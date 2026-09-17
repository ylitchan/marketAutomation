# ==================== 标准库导入 ====================
import asyncio
import atexit
import datetime
import json
import logging
import math
import os
import re
import random
import signal
import sys
import tempfile
import threading
import time
from collections import Counter
from contextvars import ContextVar
from functools import wraps
from logging.handlers import RotatingFileHandler
from urllib.parse import parse_qs, urlsplit
from dataclasses import dataclass, field, replace
from abc import ABC, abstractmethod
from decimal import ROUND_DOWN, Decimal
from enum import Enum
from typing import Literal, Mapping
from types import MappingProxyType
from zoneinfo import ZoneInfo
from uuid import uuid4

# ==================== 第三方库导入 ====================
import aiohttp
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from binance_common.configuration import (
    ConfigurationRestAPI,
    ConfigurationWebSocketStreams,
)
from binance_sdk_derivatives_trading_usds_futures.websocket_streams.websocket_streams import (
    DerivativesTradingUsdsFuturesWebSocketStreams,
)
from binance_common.errors import (
    BadRequestError,
    ForbiddenError,
    RateLimitBanError,
    RequiredError,
    TooManyRequestsError,
    UnauthorizedError,
)
from binance_sdk_derivatives_trading_portfolio_margin.derivatives_trading_portfolio_margin import (
    DerivativesTradingPortfolioMargin,
)
from binance_sdk_derivatives_trading_portfolio_margin.rest_api.models import (
    NewUmOrderPositionSideEnum,
    NewUmOrderSideEnum,
    NewUmOrderTypeEnum,
)
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

from pushplus_notifications import (
    DailyPosition,
    TradeNotification,
    format_daily_positions_notification,
    format_trade_notification,
    send_pushplus,
)


MARKET_TIMEZONE = ZoneInfo("Asia/Shanghai")


def market_time(now=None):
    if now is None:
        return datetime.datetime.now(MARKET_TIMEZONE)
    return (
        now.replace(tzinfo=MARKET_TIMEZONE)
        if now.tzinfo is None
        else now.astimezone(MARKET_TIMEZONE)
    )


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class StrategyTag(str, Enum):
    BZ = "BZ"
    BD = "BD"
    N = "N"
    BASIS = "Basis"
    DCA = "DCA"


class ActionKind(str, Enum):
    OPEN = "OPEN"
    ADD = "ADD"
    CLOSE = "CLOSE"


class OIStop(BaseModel):
    model_config = ConfigDict(frozen=True)
    open_interest: float


class VolumeStop(BaseModel):
    model_config = ConfigDict(frozen=True)
    volume: float


class Position(BaseModel):
    model_config = ConfigDict(frozen=True)
    take_profit: float
    stop_loss: float
    position_side: PositionSide
    entry_price: float
    name: str
    date: int
    strategy: tuple[StrategyTag, ...]
    tp_count: int = 0
    guard: OIStop | VolumeStop | None = None
    close_reason: str = ""
    long_short_ratio: str = ""
    basis_rate: str = ""

    @property
    def close_side(self):
        return (
            OrderSide.SELL if self.position_side == PositionSide.LONG else OrderSide.BUY
        )

    @property
    def open_side(self):
        return (
            OrderSide.BUY if self.position_side == PositionSide.LONG else OrderSide.SELL
        )

    @property
    def dca_count(self):
        return self.strategy.count(StrategyTag.DCA)


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)
    price: float
    timestamp: float
    side: OrderSide
    strategy: tuple[StrategyTag, ...]
    name: str
    bz_reference_high: float | None = None
    earliest_open_timestamp: float | None = None
    reopen_pending_date: str | None = None

    def is_reopen_cooldown_active(self, timestamp):
        return self.reopen_pending_date is not None or (
            self.earliest_open_timestamp is not None
            and timestamp < self.earliest_open_timestamp
        )


@dataclass(frozen=True)
class MarketBars:
    """市场原始响应在数据源边界转换；策略只读取带单位含义的列。"""

    times: tuple = ()
    opens: tuple[float, ...] = ()
    highs: tuple[float, ...] = ()
    lows: tuple[float, ...] = ()
    closes: tuple[float, ...] = ()
    volumes: tuple[float, ...] = ()

    def __post_init__(self):
        if (
            len(
                {
                    len(x)
                    for x in (
                        self.times,
                        self.opens,
                        self.highs,
                        self.lows,
                        self.closes,
                        self.volumes,
                    )
                }
            )
            != 1
        ):
            raise ValueError("K线列长度不一致")

    def __len__(self):
        return len(self.closes)

    @classmethod
    def from_binance(cls, rows):
        return cls(*(tuple(float(r[i]) for r in rows) for i in range(6)))

    @classmethod
    def from_frame(cls, data):
        if data.empty:
            return cls()
        times = tuple(data["date"]) if "date" in data else tuple(range(len(data)))
        return cls(
            times,
            *(
                tuple(float(v) for v in data[c])
                for c in ("open", "high", "low", "close", "volume")
            ),
        )

    @property
    def price(self):
        return self.closes[-1]

    @property
    def midprice(self):
        return (self.highs[-1] + self.lows[-1]) / 2


@dataclass(frozen=True)
class StateSnapshot:
    positions: Mapping[str, Position]
    observations: Mapping[str, Observation]


@dataclass(frozen=True)
class StatePatch:
    positions: tuple[tuple[str, Position | None], ...] = ()
    observations: tuple[tuple[str, Observation | None], ...] = ()


class TradingState:
    """唯一状态写入者；磁盘表示在此编解码，策略看不到旧JSON的混用字段。"""

    def __init__(self, market, filename):
        self.market = market
        self.filename = os.path.abspath(os.fspath(filename))
        self._loaded = False
        self._positions = {}
        self._observations = {}
        self._extras = {}
        self._pending = {}
        self.order_journal_file = self.filename + ".orders.jsonl"
        self._order_sequence = 0
        self._journal_valid_bytes = 0

    def snapshot(self):
        return StateSnapshot(
            MappingProxyType(self._positions.copy()),
            MappingProxyType(self._observations.copy()),
        )

    def apply(self, patch: StatePatch):
        # 先检查完整补丁，避免错误类型产生半份状态。
        for updates, expected in (
            (patch.positions, Position),
            (patch.observations, Observation),
        ):
            if any(
                value is not None and not isinstance(value, expected)
                for _, value in updates
            ):
                raise TypeError("状态变更类型错误")
        for target, updates in (
            (self._positions, patch.positions),
            (self._observations, patch.observations),
        ):
            for symbol, value in updates:
                if value is None:
                    target.pop(symbol, None)
                else:
                    target[symbol] = value

    def pending_order(self, symbol):
        return self._pending.get(symbol)

    def pending_symbols(self):
        return tuple(self._pending)

    def remember_order(self, order, *, durable=False):
        if durable:
            self._append_order_event(
                {"type": "pending", "order": self._encode_order(order)}
            )
        self._pending[order.intent.symbol] = order

    def resolve_order(self, symbol, *, durable=False):
        if durable:
            position = self._positions.get(symbol)
            observation = self._observations.get(symbol)
            self._append_order_event(
                {
                    "type": "resolved",
                    "symbol": symbol,
                    "position": self._encode_position(position) if position else None,
                    "observation": observation.model_dump(mode="json")
                    if observation
                    else None,
                }
            )
        self._pending.pop(symbol, None)

    def _append_order_event(self, event):
        if not self._loaded:
            raise RuntimeError("状态未加载，禁止提交真实订单")
        self._order_sequence += 1
        payload = (
            json.dumps({"sequence": self._order_sequence, **event}, ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
        mode = "r+b" if os.path.exists(self.order_journal_file) else "w+b"
        with open(self.order_journal_file, mode) as stream:
            # 上次追加失败或进程被终止时，丢弃未完整落下的尾行后再追加。
            stream.seek(self._journal_valid_bytes)
            stream.truncate()
            try:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                # 未通过持久化确认的事件不能留作下次启动的真实 pending。
                stream.seek(self._journal_valid_bytes)
                stream.truncate()
                stream.flush()
                try:
                    os.fsync(stream.fileno())
                except OSError:
                    pass
                raise
            self._journal_valid_bytes = stream.tell()

    def _load_order_journal(self):
        self._journal_valid_bytes = 0
        if not os.path.exists(self.order_journal_file):
            return
        with open(self.order_journal_file, "rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                event = json.loads(line)
                sequence = event["sequence"]
                if not isinstance(sequence, int) or sequence <= 0:
                    raise ValueError("订单日志序号无效")
                if sequence > self._order_sequence:
                    if event["type"] == "pending":
                        self.remember_order(self._decode_pending(event["order"]))
                    elif event["type"] == "resolved":
                        symbol = event["symbol"]
                        self.apply(
                            StatePatch(
                                positions=(
                                    (
                                        symbol,
                                        self.decode_position(event["position"])
                                        if event["position"]
                                        else None,
                                    ),
                                ),
                                observations=(
                                    (
                                        symbol,
                                        Observation.model_validate(event["observation"])
                                        if event["observation"]
                                        else None,
                                    ),
                                ),
                            )
                        )
                        self.resolve_order(symbol)
                    else:
                        raise ValueError("订单日志事件无效")
                    self._order_sequence = sequence
                self._journal_valid_bytes += len(line)

    def decode_position(self, row):
        data = row.copy()
        threshold = data.pop("stop_guard_threshold", 0.0)
        data.pop("close_side", None)
        if self.market == "AUTOBN":
            data["guard"] = OIStop(open_interest=threshold)
        else:
            data["guard"] = VolumeStop(volume=threshold)
        return Position.model_validate(data)

    def load(self):
        with open(self.filename, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or any(
            not isinstance(data.get(k, {}), dict) for k in ("POSITIONS", "OBSERVATIONS")
        ):
            raise ValueError("状态文件格式错误")
        positions = {
            k: self.decode_position(v) for k, v in data.get("POSITIONS", {}).items()
        }
        observations = {
            k: Observation.model_validate(v)
            for k, v in data.get("OBSERVATIONS", {}).items()
        }
        raw_pending = data.get("PENDING_ORDERS", {})
        if not isinstance(raw_pending, dict):
            raise ValueError("状态文件待确认订单格式错误")
        pending = {k: self._decode_pending(v) for k, v in raw_pending.items()}
        self._positions, self._observations = positions, observations
        self._pending = pending
        self._extras = {
            k: v
            for k, v in data.items()
            if k
            not in (
                "POSITIONS",
                "OBSERVATIONS",
                "PENDING_ORDERS",
                "ORDER_JOURNAL_SEQUENCE",
            )
        }
        self._order_sequence = int(data.get("ORDER_JOURNAL_SEQUENCE", 0))
        self._load_order_journal()
        self._loaded = True

    def _decode_pending(self, row):
        raw = row.copy()
        intent = raw.pop("intent").copy()
        intent["kind"] = ActionKind(intent["kind"])
        position = intent["position"].copy()
        guard = position.pop("guard", None)
        if guard:
            position["guard"] = (
                OIStop.model_validate(guard)
                if "open_interest" in guard
                else VolumeStop.model_validate(guard)
            )
        intent["position"] = Position.model_validate(position)
        if intent.get("observation"):
            intent["observation"] = Observation.model_validate(intent["observation"])
        return PreparedOrder(TradeIntent(**intent), **raw)

    @staticmethod
    def _encode_position(position):
        row = position.model_dump(mode="json", exclude={"guard"})
        row["close_side"] = position.close_side.value
        guard = position.guard
        row["stop_guard_threshold"] = (
            guard.open_interest
            if isinstance(guard, OIStop)
            else guard.volume
            if isinstance(guard, VolumeStop)
            else 0.0
        )
        return row

    @staticmethod
    def _encode_order(order):
        row = vars(order).copy()
        intent = vars(order.intent).copy()
        intent["kind"] = order.intent.kind.value
        intent["position"] = order.intent.position.model_dump(mode="json")
        intent["observation"] = (
            order.intent.observation.model_dump(mode="json")
            if order.intent.observation
            else None
        )
        row["intent"] = intent
        return row

    def serialize(self):
        data = {
            **self._extras,
            "POSITIONS": {
                k: self._encode_position(v) for k, v in self._positions.items()
            },
            "OBSERVATIONS": {
                k: v.model_dump(mode="json") for k, v in self._observations.items()
            },
        }
        if self._order_sequence:
            data["ORDER_JOURNAL_SEQUENCE"] = self._order_sequence
        if self._pending:
            data["PENDING_ORDERS"] = {
                k: self._encode_order(v) for k, v in self._pending.items()
            }
        return data

    def save(self):
        if not self._loaded:
            return
        payload = json.dumps(self.serialize(), ensure_ascii=False, indent=4)
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=os.path.dirname(self.filename),
                prefix=os.path.basename(self.filename) + ".",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_name = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.filename)
            # 快照带有日志序号；这里中断也不会重复重放旧事件覆盖更新后的轨道。
            if os.path.exists(self.order_journal_file):
                os.remove(self.order_journal_file)
            self._journal_valid_bytes = 0
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.remove(temporary_name)


@dataclass(frozen=True)
class TradeIntent:
    kind: ActionKind
    symbol: str
    position: Position
    price: float
    atr: float = 0
    ratio: float = 1.0
    take_profit_stage: Literal["first", "regular"] | None = None
    observation: Observation | None = None
    signal_return: float = 0


@dataclass(frozen=True)
class Decision:
    patch: StatePatch = field(default_factory=StatePatch)
    intent: TradeIntent | None = None
    proceed: bool = True


@dataclass(frozen=True)
class PreparedOrder:
    intent: TradeIntent
    quantity: float
    price: float
    entry_price: float
    ratio: float = 1
    leverage: int = 1
    account_equity: float = 0
    notional: float = 0
    no_position: bool = False
    client_order_id: str = ""


class ExecutionStatus(str, Enum):
    FILLED = "FILLED"
    FAILED = "FAILED"
    NO_POSITION = "NO_POSITION"
    UNKNOWN = "UNKNOWN"


class ScanOutcome(str, Enum):
    PROCESSED = "processed"
    SKIPPED = "skipped"
    NO_DATA = "no_data"
    PENDING = "pending"
    FAILED = "failed"


@dataclass(frozen=True)
class ExecutionResult:
    status: ExecutionStatus
    quantity: float = 0
    price: float = 0
    entry_price: float = 0
    ratio: float = 1
    original_quantity: str | float = 0
    average_price: float | None = None
    remaining_quantity: float | None = None

    @property
    def filled(self):
        return self.status == ExecutionStatus.FILLED


@dataclass
class MarketContext:
    now: datetime.datetime
    start_date: str | None = None
    end_date: str | None = None
    exit_only: bool = False
    calendar_request: asyncio.Task | None = None


@dataclass
class ScanCache:
    window_id: int | None = None
    snapshot: tuple[str, ...] = ()


@dataclass
class BinanceCache:
    recovery_symbols: set[str] = field(default_factory=set)
    universe: tuple[str, ...] = ()
    symbols_info: dict = field(default_factory=dict)
    lsr_1h: dict = field(default_factory=dict)
    lsr_5m: dict = field(default_factory=dict)
    oi_history: dict = field(default_factory=dict)
    oi_5m: dict = field(default_factory=dict)
    basis_rate: dict = field(default_factory=dict)
    bootstrap_attempted: bool = False
    positions: tuple = ()
    positions_at: datetime.datetime | None = None
    kline_history: dict = field(default_factory=dict)


@dataclass
class AshareCache:
    history: dict = field(default_factory=dict)
    calendar: pd.Series | None = None
    calendar_loaded_on: datetime.date | None = None
    trading_dates: list[str] = field(default_factory=list)


def format_strategy_tags(strategies):
    counts = {}
    for strategy in strategies:
        value = strategy.value if isinstance(strategy, Enum) else str(strategy)
        counts[value] = counts.get(value, 0) + 1
    return ",".join(f"{v}*{count}" if count > 1 else v for v, count in counts.items())


def chebyshev_result(mean, std, value):
    deviation = value - mean
    if std == 0:
        k = float("inf") if deviation != 0 else 0.0
        upper = 0.0
    else:
        k = abs(deviation) / std
        upper = 1.0 if k <= 1 else 1 / k**2
    return {
        "mean": mean,
        "std": std,
        "k": k,
        "chebyshev_upper_bound": upper,
        "min_probability_in_range": 1 - upper,
        "deviation": deviation,
    }


def sample_probability(data, value, *, pandas_sample=False):
    if not len(data):
        raise ValueError("样本不能为空")
    if pandas_sample:
        series = pd.Series(data)
        mean = float(series.mean())
        std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    else:
        mean = sum(data) / len(data)
        std = (
            (sum((x - mean) ** 2 for x in data) / (len(data) - 1)) ** 0.5
            if len(data) > 1
            else 0.0
        )
    return chebyshev_result(mean, std, value)


class CloseRecordManager:
    """
    由主入口创建并注入两个策略的平仓记录管理器

    功能：将AUTOBN和AUTOA的平仓记录分别写入Excel的不同sheet
    特点：
        - 线程安全：使用锁保护写入操作
        - 自动创建：文件不存在时自动创建
        - 分sheet存储：AUTOBN写入"币安期货"，AUTOA写入"A股"
        - 追加模式：每次平仓追加一行记录
    """

    def __init__(self, filename=None):
        self._excel_file = filename or os.path.join(
            os.path.dirname(__file__), "close_records.xlsx"
        )
        self._lock = threading.Lock()
        self._logger = logging.getLogger("CloseRecordManager")
        self._pending_records = []
        self._batch_lock = None
        self._batch_lock_loop = None

    # Sheet名称映射
    SHEET_NAMES = MappingProxyType(
        {
            "AUTOBN": "币安期货",
            "AUTOA": "A股",
        }
    )

    # Excel列定义
    COLUMNS = (
        "平仓时间",
        "交易品种",  # symbol 或 股票代码
        "持仓方向",  # LONG/SHORT
        "开仓价格",
        "平仓价格",
        "平仓数量",
        "平仓盈亏(USDT/CNY)",
        "平仓收益率",
        "平仓比例",
        "策略标签",  # Supertrend/BZ/BD 等
        "平仓依据",
        "多空比",
        "基差率",
        "订单编号",
    )

    def _record_close_batch(self, records):
        with self._lock:
            existing_sheets = {}
            if os.path.exists(self._excel_file):
                try:
                    with pd.ExcelFile(self._excel_file, engine="openpyxl") as xls:
                        for name in xls.sheet_names:
                            existing_sheets[name] = pd.read_excel(xls, sheet_name=name)
                except Exception:
                    pass

            records_by_sheet = {}
            for record in records:
                sheet_name = self.SHEET_NAMES.get(record["source"], record["source"])
                records_by_sheet.setdefault(sheet_name, []).append(
                    {key: value for key, value in record.items() if key != "source"}
                )

            for sheet_name, sheet_records in records_by_sheet.items():
                df = existing_sheets.get(sheet_name, pd.DataFrame(columns=self.COLUMNS))
                existing_sheets[sheet_name] = pd.concat(
                    [df, pd.DataFrame(sheet_records)], ignore_index=True
                )
                combined = existing_sheets[sheet_name]
                if "订单编号" in combined:
                    order_ids = combined["订单编号"]
                    duplicate = (
                        order_ids.notna() & order_ids.ne("") & order_ids.duplicated()
                    )
                    existing_sheets[sheet_name] = combined[~duplicate]

            with pd.ExcelWriter(self._excel_file, engine="openpyxl") as writer:
                for name, data in existing_sheets.items():
                    data.to_excel(writer, sheet_name=name, index=False)

    def enqueue_close(
        self,
        source: str,
        symbol: str,
        position_side: str,
        entry_price: float,
        close_price: float,
        close_amount: float,
        realized_pnl: float,
        pnl_percent: float,
        close_ratio: float = 1.0,
        strategy_tag: str = "",
        close_reason: str = "",
        long_short_ratio: str = "",
        basis_rate: str = "",
        execution_id: str = "",
    ):
        record = {
            "source": source,
            "平仓时间": datetime.datetime.now(MARKET_TIMEZONE).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "交易品种": symbol,
            "持仓方向": position_side,
            "开仓价格": entry_price,
            "平仓价格": close_price,
            "平仓数量": close_amount,
            "平仓盈亏(USDT/CNY)": realized_pnl,
            "平仓收益率": f"{pnl_percent:.2%}",
            "平仓比例": f"{close_ratio:.2%}",
            "策略标签": strategy_tag,
            "平仓依据": close_reason,
            "多空比": long_short_ratio,
            "基差率": basis_rate,
            "订单编号": execution_id,
        }
        self._pending_records.append(record)
        return record

    async def flush_pending_records(self):
        if not self._pending_records:
            return
        current_loop = asyncio.get_running_loop()
        if self._batch_lock is None or self._batch_lock_loop is not current_loop:
            self._batch_lock = asyncio.Lock()
            self._batch_lock_loop = current_loop
        async with self._batch_lock:
            records = self._pending_records[:]
            if not records:
                return
            # 线程写入无法被 asyncio 取消；保持锁直到写入结果确定，避免重试重复记账。
            write_task = asyncio.create_task(
                asyncio.to_thread(self._record_close_batch, records)
            )
            try:
                await asyncio.shield(write_task)
            except asyncio.CancelledError:
                try:
                    await write_task
                except Exception:
                    self._logger.exception("平仓记录写入失败，保留待重试")
                else:
                    del self._pending_records[: len(records)]
                raise
            except Exception:
                self._logger.exception("平仓记录写入失败，保留待重试")
                return
            del self._pending_records[: len(records)]
            self._logger.info(f"平仓记录保存完成，共{len(records)}条")


class HttpSession:
    def __init__(self, timeout=10):
        self.timeout = timeout
        self._session = None

    async def get(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None


class Notifications:
    def __init__(self, http, logger):
        self.http, self.logger = http, logger

    async def send(self, message, *, channel="pushplus"):
        self.logger.info(
            f"发送消息: {message.content if isinstance(message, TradeNotification) else message}"
        )
        try:
            sent = await asyncio.wait_for(
                send_pushplus(message, channel=channel), self.http.timeout
            )
            if not sent:
                self.logger.warning("未配置 PUSHPLUS_TOKEN")
        except Exception:
            self.logger.exception(f"{channel}消息发送失败")


_BINANCE_REQUEST_SCOPE = ContextVar("binance_request_scope", default="other")


class BinanceRequestDiagnostics:
    """临时请求观测：只记白名单标签和数值，绝不改变请求/重试/交易行为。"""

    ENDPOINTS = MappingProxyType(
        {
            "/fapi/v1/klines": "klines",
            "/futures/data/openInterestHist": "oi",
            "/futures/data/globalLongShortAccountRatio": "lsr",
            "/fapi/v1/premiumIndex": "mark_price",
            "/fapi/v1/exchangeInfo": "exchange_info",
            "/papi/v1/account": "account",
            "/papi/v1/um/positionRisk": "positions",
            "/papi/v1/um/leverage": "leverage",
            "/papi/v1/um/order": "order",
        }
    )
    PERIODS = frozenset(("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"))
    SCOPES = frozenset(("held", "observed", "unobserved", "pending", "other"))

    def __init__(self, *, logger=None):
        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger("BinanceRequests")
            if not self.logger.handlers:
                handler = RotatingFileHandler(
                    os.path.join(os.path.dirname(__file__), "binance_requests.log"),
                    maxBytes=5 * 1024 * 1024,
                    backupCount=2,
                    encoding="utf-8",
                    delay=True,
                )
                handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
                self.logger.propagate = False
        self._lock = threading.Lock()
        self._buckets = {}
        self._headers = {}
        self._last_report = time.monotonic()
        self._started_at = self._last_report
        self._last_limit = {}
        self._local = threading.local()
        self._sessions = set()

    def _prune(self, now):
        cutoff = int(now) - 300
        for second in tuple(self._buckets):
            if second <= cutoff:
                del self._buckets[second]
        for key, (timestamp, _) in tuple(self._headers.items()):
            if timestamp <= now - 300:
                del self._headers[key]

    def note(self, kind, endpoint, period="", detail=""):
        try:
            scope = _BINANCE_REQUEST_SCOPE.get()
            if scope not in self.SCOPES:
                scope = "other"
            key = ":".join((kind, endpoint, period, scope, detail))
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                self._buckets.setdefault(int(now), Counter())[key] += 1
        except Exception:
            pass  # 诊断故障不能改变业务结果。

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            minute, five = Counter(), Counter()
            for second, counts in self._buckets.items():
                five.update(counts)
                if second > int(now) - 60:
                    minute.update(counts)
            return {
                "pid": os.getpid(),
                "time": market_time().isoformat(),
                "observed_seconds": round(max(0, now - self._started_at), 1),
                "http_1m": sum(v for k, v in minute.items() if k.startswith("http:")),
                "http_5m": sum(v for k, v in five.items() if k.startswith("http:")),
                "sdk_calls_5m": sum(v for k, v in five.items() if k.startswith("sdk:")),
                "extra_sends_5m": sum(
                    v for k, v in five.items() if k.startswith("extra_send:")
                ),
                "events_1m": dict(minute),
                "events_5m": dict(five),
                "server_headers": {
                    k: {"value": v, "age_seconds": round(now - t, 1)}
                    for k, (t, v) in self._headers.items()
                },
            }

    def report(self, *, limit=None, force=False):
        try:
            with self._lock:
                now = time.monotonic()
                status = limit["http_status"] if limit else None
                last_limit = self._last_limit.get(status)
                first_limit = limit is not None and (
                    last_limit is None or now - last_limit >= 300
                )
                if not force and not first_limit and now - self._last_report < 60:
                    return
                if first_limit:
                    self._last_limit[status] = now
                self._last_report = now
            payload = self.snapshot()
            payload["event"] = "rate_limit" if first_limit else "minute_summary"
            if limit is not None:
                payload["trigger"] = limit
            self.logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        except Exception:
            try:
                logging.getLogger("AutoTrade").warning(
                    "币安诊断汇总记录失败，交易流程继续"
                )
            except Exception:
                pass

    def invoke(self, method, *args, **kwargs):
        previous = getattr(self._local, "attempts", None)
        self._local.attempts = 0
        self.note("sdk", "call")
        try:
            return method(*args, **kwargs)
        finally:
            self._local.attempts = previous
            self.report()

    def attach(self, session, service):
        if id(session) in self._sessions:
            return
        self._sessions.add(id(session))
        original = session.send

        @wraps(original)
        def send(request, **kwargs):
            endpoint, period, symbol = "other", "", ""
            method = (
                request.method
                if request.method in ("GET", "POST", "PUT", "DELETE", "HEAD")
                else "OTHER"
            )
            try:
                url = urlsplit(request.url)
                endpoint = self.ENDPOINTS.get(url.path, "other")
                parameters = parse_qs(url.query)
                value = parameters.get("period", parameters.get("interval", [""]))[0]
                period = value if value in self.PERIODS else ""
                value = parameters.get("symbol", [""])[0]
                symbol = value if re.fullmatch(r"[A-Z0-9_]{1,40}", value) else ""
                attempt = getattr(self._local, "attempts", None)
                if attempt is not None:
                    self._local.attempts = attempt + 1
                    if attempt:
                        self.note("extra_send", endpoint, period, service)
                self.note("http", endpoint, period, service + "/" + method)
            except Exception:
                pass
            try:
                response = original(request, **kwargs)
            except Exception:
                self.note("network_error", endpoint, period, service)
                self.report()
                raise
            try:
                status = int(response.status_code)
                self.note(
                    "status",
                    endpoint,
                    period,
                    str(status) if 100 <= status <= 599 else "other",
                )
                selected = {}
                for key, value in response.headers.items():
                    key = key.lower()
                    if re.fullmatch(
                        r"(?:x-mbx|x-sapi)-used-(?:weight|ip-weight)(?:-\d+[smhd])?|x-mbx-order-count-\d+[smhd]|retry-after",
                        key,
                    ):
                        if re.fullmatch(r"\d{1,16}", str(value)):
                            selected[key] = int(value)
                with self._lock:
                    for key, value in selected.items():
                        self._headers[service + "/" + key] = (time.monotonic(), value)
                trigger = {
                    "service": service,
                    "endpoint": endpoint,
                    "period": period,
                    "symbol": symbol,
                    "http_status": status,
                    "headers": selected,
                }
                self.report(limit=trigger if status in (429, 418) else None)
            except Exception:
                pass
            return response

        session.send = send


class BinanceGateway:
    API_TIMEOUT_SECONDS = 15

    def __init__(
        self, *, market_client=None, papi_client=None, api_key=None, api_secret=None
    ):
        if market_client is None or papi_client is None:
            api_key = api_key or os.getenv("BINANCE_API_KEY")
            api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
            if not api_key or not api_secret:
                raise ValueError("币安API密钥未配置")
            market_client = DerivativesTradingUsdsFutures(
                config_rest_api=ConfigurationRestAPI(
                    api_key=api_key, api_secret=api_secret, timeout=5000
                )
            )
            papi_client = DerivativesTradingPortfolioMargin(
                config_rest_api=ConfigurationRestAPI(
                    api_key=api_key, api_secret=api_secret, timeout=5000
                )
            )
            adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
            market_client.rest_api._session.mount("https://", adapter)
            market_client.rest_api._session.mount("http://", adapter)
        self.market_client, self.papi_client = market_client, papi_client
        self._semaphore = asyncio.Semaphore(8)
        self.diagnostics = BinanceRequestDiagnostics()
        for client, service in ((market_client, "fapi"), (papi_client, "papi")):
            self.diagnostics.attach(client.rest_api._session, service)

    @staticmethod
    def unwrap(data):
        if hasattr(data, "data") and callable(data.data):
            data = data.data()
        if isinstance(data, list):
            return [BinanceGateway.unwrap(x) for x in data]
        if isinstance(data, dict):
            return data
        if hasattr(data, "to_dict"):
            return data.to_dict()
        if hasattr(data, "model_dump"):
            return data.model_dump(by_alias=True, exclude_none=True)
        return data

    async def call(self, method, *args, **kwargs):
        async with self._semaphore:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.diagnostics.invoke, method, *args, **kwargs),
                self.API_TIMEOUT_SECONDS,
            )
            return self.unwrap(result)

    async def account(self):
        info = await self.call(
            self.papi_client.rest_api.account_information, recv_window=60000
        )
        return {
            "available": float(info.get("totalAvailableBalance", 0) or 0),
            "equity": float(info.get("accountEquity", 0) or 0),
            "maintenance": float(info.get("accountMaintMargin", 0) or 0),
        }

    async def positions(self, symbol=None, position_side=None):
        args = {"symbol": symbol, "recv_window": 60000} if symbol else {}
        rows = await self.call(
            self.papi_client.rest_api.query_um_position_information, **args
        )
        if not isinstance(rows, list) or any(
            not isinstance(row, dict)
            or not {"symbol", "positionSide", "positionAmt"}.issubset(row)
            for row in rows
        ):
            raise ValueError("币安持仓响应不完整，不能当作空仓")
        return [
            row
            for row in rows
            if float(row.get("positionAmt", 0) or 0) != 0
            and (position_side is None or row.get("positionSide") == position_side)
        ]

    async def position_amount(self, symbol, position_side=None):
        try:
            rows = await self.positions(symbol, position_side)
            for row in reversed(rows):
                if row.get("symbol") == symbol:
                    return abs(float(row["positionAmt"])), float(
                        row.get("entryPrice", 0) or 0
                    )
            return 0.0, 0.0
        except Exception:
            return None

    async def submit(self, symbol, side, position_side, amount, client_order_id):
        return await self.call(
            self.papi_client.rest_api.new_um_order,
            symbol=symbol,
            side=NewUmOrderSideEnum(side),
            type=NewUmOrderTypeEnum("MARKET"),
            quantity=amount,
            position_side=NewUmOrderPositionSideEnum(position_side),
            new_client_order_id=client_order_id,
            recv_window=60000,
        )

    async def query_order(self, symbol, client_order_id):
        return await self.call(
            self.papi_client.rest_api.query_um_order,
            symbol=symbol,
            orig_client_order_id=client_order_id,
            recv_window=60000,
        )


class BinanceKlineStream(DerivativesTradingUsdsFuturesWebSocketStreams):
    """官方SDK连接/接收/回执能力的批次适配；不使用SDK逐币订阅及全局订阅表。"""

    def __init__(self, on_message):
        super().__init__(
            ConfigurationWebSocketStreams(stream_url="wss://fstream.binance.com/stream")
        )
        self._on_kline = on_message
        self._background = set()
        self._connection = None
        self._intentional_close = False
        self._command_lock = asyncio.Lock()
        self.disconnected = asyncio.Event()

    async def open(self):
        # 仅建立market路由；public/private不是日K所需连接。
        await asyncio.wait_for(
            self.connect(
                self.configuration.stream_url, self.configuration, url_paths=["market"]
            ),
            8,
        )
        if len(self.connections) != 1:
            raise ConnectionError("SDK未建立market连接")
        self._connection = self.connections[0]

    async def receive_loop(self, connection):
        task = asyncio.current_task()
        self._background.add(task)
        try:
            await super().receive_loop(connection)
        except Exception:
            logging.getLogger("AutoTrade.WS").exception("官方SDK行情接收失败")
        finally:
            self._background.discard(task)
            if not self._intentional_close:
                self.disconnected.set()

    async def schedule_reconnect(
        self, connection, configuration, delay, close_old_connection=True
    ):
        # SDK创建的轮换任务也受适配器关闭管理。真正重连由唯一市场数据管理者执行。
        task = asyncio.current_task()
        self._background.add(task)
        try:
            await asyncio.sleep(delay)
            if not self._intentional_close:
                self.disconnected.set()
        finally:
            self._background.discard(task)

    async def command(self, method, streams=None):
        async with self._command_lock:
            connection = self._connection
            if connection is None or self.disconnected.is_set():
                raise ConnectionError("SDK行情连接不可用")
            request_id = uuid4().hex
            payload = {"method": method, "id": request_id}
            if streams is not None:
                payload["params"] = streams
            try:
                future = await super().send_message(payload, connection)
                response = await asyncio.wait_for(future, 3)
                if (
                    response.get("id") != request_id
                    or "result" not in response
                    or response.get("code") is not None
                ):
                    raise ValueError("WS控制请求失败")
                return response["result"]
            except BaseException:
                self.disconnected.set()
                raise
            finally:
                connection.pending_request.pop(request_id, None)

    async def subscribe_batch(self, symbols):
        streams = [f"{symbol.lower()}@kline_1d" for symbol in symbols]
        if streams:
            # 这些回调表是SDK连接对象的接收契约，封装在此SDK子类内；业务层不访问SDK状态。
            for stream in streams:
                self._connection.stream_callback_map[stream] = [self._on_kline]
                self._connection.response_types[stream] = None
            if await self.command("SUBSCRIBE", streams) is not None:
                raise ValueError("WS订阅回执无效")

    async def unsubscribe_batch(self, symbols):
        streams = [f"{symbol.lower()}@kline_1d" for symbol in symbols]
        if streams:
            if await self.command("UNSUBSCRIBE", streams) is not None:
                raise ValueError("WS退订回执无效")
            for stream in streams:
                self._connection.stream_callback_map.pop(stream, None)
                self._connection.response_types.pop(stream, None)

    async def finish(self):
        self._intentional_close = True
        await super().close_connection()
        # init_connection通过create_task创建SDK收包/轮换任务，让它们注册后统一回收。
        await asyncio.sleep(0)
        tasks = tuple(
            task for task in self._background if task is not asyncio.current_task()
        )
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._background.clear()
        self._connection = None


class InsufficientKlineHistory(ValueError):
    """已有历史真实不足，继续等待本轮推送不能补齐。"""


class MarketData(ABC):
    @abstractmethod
    async def fetch_bars(self, symbol: str, context: MarketContext) -> MarketBars: ...

    async def begin_slot(self, symbols, deadline):
        return tuple(symbols)

    async def end_slot(self):
        return None

    async def wait_ready(self, symbol):
        return True

    def complete_slot_symbol(self, symbol):
        return None

    async def close(self):
        return None


class BinanceMarketData(MarketData):
    KLINE_LIMIT = 30
    LONG_SHORT_RATIO_LIMIT = 30
    OI_QUERY_LIMIT = 30
    FIVE_MIN_PERIOD_MS = 300_000
    BASIS_RATE_CACHE_TTL = 300
    KLINE_INTERVAL_MS = 86_400_000

    def __init__(self, gateway, logger):
        self.gateway, self.logger = gateway, logger
        self.cache = BinanceCache()
        self._stream = None
        self._ws_task = None
        self._closing = False
        self._slot_lock = asyncio.Lock()
        self._slot_symbols = set()
        self._subscribed_symbols = set()
        self._slot_events = {}
        self._latest_klines = {}
        self._slot_deadline = None
        self._accept_after_ms = 0
        self._connection_generation = 0
        self._active_generation = None
        self._disconnect_notifier = None
        self._disconnect_notified = False
        self._reconnect_delay = 1.0
        self._notice_tasks = set()

    def set_disconnect_notifier(self, notifier):
        self._disconnect_notifier = notifier

    async def begin_slot(self, symbols, deadline):
        if self._closing or self._slot_deadline is not None:
            raise RuntimeError("行情批次生命周期冲突")
        self._slot_symbols.update(symbols)
        self._slot_events = {symbol: asyncio.Event() for symbol in self._slot_symbols}
        self._slot_deadline = deadline
        if not self._slot_symbols:
            return ()
        async with self._slot_lock:
            if self._stream is not None and not self._stream.disconnected.is_set():
                try:
                    await self._subscribe_active()
                except Exception as error:
                    self._connection_failed(error)
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._ws_loop())
        return tuple(sorted(self._slot_symbols))

    async def _subscribe_active(self):
        added = self._slot_symbols - self._subscribed_symbols
        if added:
            await self._stream.subscribe_batch(sorted(added))
            self._subscribed_symbols.update(added)
            self.logger.info(
                f"WS新增订阅:{len(added)} 保留订阅:{len(self._subscribed_symbols)}"
            )
        # 首次连接/重连只接受确认订阅后的事件，已有连接增量订阅不打断其他行情。
        if self._active_generation is None:
            self._accept_after_ms = int(time.time() * 1000)
            self._active_generation = self._connection_generation

    async def end_slot(self):
        self._slot_events.clear()
        self._slot_deadline = None
        async with self._slot_lock:
            completed = sorted(self._subscribed_symbols - self._slot_symbols)
            if completed and self._stream is not None:
                try:
                    await self._stream.unsubscribe_batch(completed)
                    self._subscribed_symbols.difference_update(completed)
                    self.logger.info(
                        f"WS检测结束退订:{len(completed)} 待处理:{len(self._slot_symbols)}"
                    )
                except Exception as error:
                    self._connection_failed(error)
                    await self._stream.finish()
            elif self._slot_symbols:
                self.logger.info(f"WS继续保留待处理标的:{len(self._slot_symbols)}")

    def complete_slot_symbol(self, symbol):
        self._slot_symbols.discard(symbol)
        self._latest_klines.pop(symbol, None)

    def _connection_failed(self, error):
        self._active_generation = None
        self._latest_klines.clear()
        for event in self._slot_events.values():
            event.clear()
        if self._stream is not None:
            self._stream.disconnected.set()
        self.logger.warning(f"币安K线WS连接异常:{type(error).__name__}，准备重连")
        if not self._closing and not self._disconnect_notified:
            self._disconnect_notified = True
            if self._disconnect_notifier is not None:
                task = asyncio.create_task(
                    self._notify_disconnect(type(error).__name__)
                )
                self._notice_tasks.add(task)
                task.add_done_callback(self._notice_tasks.discard)

    async def _notify_disconnect(self, reason):
        try:
            await asyncio.wait_for(
                self._disconnect_notifier(
                    f"币安K线WS连接异常，正在重连\n原因:{reason}\n时间:{market_time().isoformat()}"
                ),
                10,
            )
        except Exception:
            self.logger.exception("WS断连PushPlus通知失败")

    async def _ws_loop(self):
        while not self._closing:
            try:
                if self._stream is None or self._stream.disconnected.is_set():
                    async with self._slot_lock:
                        if self._stream is not None:
                            await self._stream.finish()
                        self._connection_generation += 1
                        generation = self._connection_generation
                        self._stream = BinanceKlineStream(
                            lambda data, generation=generation: self._handle_ws_message(
                                data, generation
                            )
                        )
                        await self._stream.open()
                        self._subscribed_symbols.clear()
                        await self._subscribe_active()
                    self.logger.info("币安K线WS连接已建立（官方SDK market）")
                # SDK处理Ping/Pong；仅在接收结束、异常或轮换时恢复连接。
                await self._stream.disconnected.wait()
                raise ConnectionError("WS连接关闭或轮换")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._connection_failed(error)
                await asyncio.sleep(
                    random.uniform(self._reconnect_delay * 0.5, self._reconnect_delay)
                )
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)

    async def close(self):
        self._closing = True
        self._active_generation = None
        if self._ws_task is not None:
            self._ws_task.cancel()
            await asyncio.gather(self._ws_task, return_exceptions=True)
            self._ws_task = None
        if self._stream is not None:
            await self._stream.finish()
            self._stream = None
        for task in tuple(self._notice_tasks):
            task.cancel()
        await asyncio.gather(*self._notice_tasks, return_exceptions=True)
        self._notice_tasks.clear()
        self._slot_deadline = None
        self._slot_symbols.clear()
        self._latest_klines.clear()
        self._slot_events.clear()
        self._subscribed_symbols.clear()

    @classmethod
    def _valid_ohlcv(cls, row):
        try:
            o, high, low, c, v = map(float, row[1:6])
            return (
                all(math.isfinite(x) for x in (o, high, low, c, v))
                and 0 < low <= min(o, c) <= max(o, c) <= high
                and v >= 0
            )
        except (TypeError, ValueError, OverflowError):
            return False

    def _handle_ws_message(self, payload, generation):
        if generation != self._active_generation or self._closing:
            return
        payload = payload.get("data", payload) if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or payload.get("e") != "kline":
            return
        k = payload.get("k")
        if not isinstance(k, dict):
            return
        symbol = k.get("s")
        if (
            symbol not in self._subscribed_symbols
            or symbol not in self._slot_symbols
            or payload.get("s") != symbol
            or k.get("i") != "1d"
        ):
            return
        try:
            now_ms = int(time.time() * 1000)
            d = now_ms // self.KLINE_INTERVAL_MS * self.KLINE_INTERVAL_MS
            event_ms = payload["E"]
            if (
                type(event_ms) is not int
                or type(k["t"]) is not int
                or k["t"] != d
                or k.get("T") != d + self.KLINE_INTERVAL_MS - 1
                or k.get("x") is not False
            ):
                return
            if event_ms <= self._accept_after_ms:
                return
            row = (d, k["o"], k["h"], k["l"], k["c"], k["v"])
            if not self._valid_ohlcv(row):
                return
            previous = self._latest_klines.get(symbol)
            if previous is not None and event_ms <= previous[0]:
                return
            self._latest_klines[symbol] = (event_ms, row)
            self._reconnect_delay = 1.0
            event = self._slot_events.get(symbol)
            if event is not None:
                event.set()
            if previous is None:
                self.logger.debug(f"WS首条行情收到 symbol={symbol} open_time={d}")
            if self._disconnect_notified:
                self._disconnect_notified = False
                self.logger.info("币安K线WS行情已恢复")
        except (KeyError, TypeError, ValueError, OverflowError):
            return

    def _latest_bar(self, symbol):
        value = self._latest_klines.get(symbol)
        now_ms = int(time.time() * 1000)
        if (
            value is None
            or self._active_generation is None
            or self._slot_deadline is None
            or time.monotonic() >= self._slot_deadline
            or value[1][0] != now_ms // self.KLINE_INTERVAL_MS * self.KLINE_INTERVAL_MS
        ):
            return None
        return value[1]

    async def wait_ready(self, symbol):
        event = self._slot_events.get(symbol)
        if event is None:
            return False
        try:
            async with asyncio.timeout(max(0, self._slot_deadline - time.monotonic())):
                while self._latest_bar(symbol) is None:
                    event.clear()
                    await event.wait()
            return True
        except asyncio.TimeoutError:
            self.logger.info(f"WS本轮尚无行情 symbol={symbol}，保留订阅待下轮检测")
            return False

    def universe(self):
        return self.cache.universe

    def precision(self, symbol):
        return self.cache.symbols_info.get(symbol, {}).get("quantityPrecision")

    async def account_positions(self, now):
        # 日报和紧随其后的刷新处于同一轮、同一把锁内，复用已确认的账户响应。
        if self.cache.positions_at != now:
            rows = await self.gateway.positions()
            self.cache.positions = tuple(MappingProxyType(row.copy()) for row in rows)
            self.cache.positions_at = now
        return self.cache.positions

    def positions_snapshot(self, now):
        return self.cache.positions if self.cache.positions_at == now else None

    def recovery_symbols(self):
        return tuple(self.cache.recovery_symbols)

    def require_position_recovery(self, symbol):
        self.cache.recovery_symbols.add(symbol)

    def resolve_position_recovery(self, symbol):
        self.cache.recovery_symbols.discard(symbol)

    async def bootstrap(self, now):
        if self.cache.bootstrap_attempted:
            return False
        self.cache.bootstrap_attempted = True
        if self.cache.universe:
            return False
        try:
            return await self.refresh(now)
        except Exception:
            self.logger.exception("首次交易对获取失败，等待每日刷新")
            return False

    async def refresh(self, now):
        response, positions = await asyncio.gather(
            self.gateway.call(self.gateway.market_client.rest_api.exchange_information),
            self.account_positions(now),
            return_exceptions=True,
        )
        if isinstance(positions, BaseException):
            self.logger.warning(
                f"实际持仓核对失败:{type(positions).__name__}，保留本地状态"
            )
        if isinstance(response, BaseException):
            self.logger.warning(
                f"交易对元数据刷新失败:{type(response).__name__}，保留旧池"
            )
            return False
        metadata = {}
        try:
            if not isinstance(response, dict) or not isinstance(
                response.get("symbols"), list
            ):
                raise ValueError("交易对元数据结构无效")
            for row in response["symbols"]:
                if row["quoteAsset"] != "USDT" or row["status"] != "TRADING":
                    continue
                precision = row["quantityPrecision"]
                if not isinstance(precision, int) or precision < 0:
                    raise ValueError("交易对数量精度无效")
                metadata[row["symbol"]] = {
                    "quotePrecision": row["quotePrecision"],
                    "quantityPrecision": Decimal("1")
                    if precision == 0
                    else Decimal(f"0.{'1' * precision}"),
                }
        except (KeyError, TypeError, ValueError) as error:
            self.logger.warning(
                f"交易对元数据解析失败:{type(error).__name__}，保留旧池"
            )
            return False
        if not metadata:
            return False
        self.cache.symbols_info = metadata
        self.cache.universe = tuple(metadata)
        self.cache.bootstrap_attempted = True
        return True

    async def fetch_bars(self, symbol, context):
        current = self._latest_bar(symbol)
        if current is None:
            return MarketBars()
        d = current[0]
        try:
            history = self.cache.kline_history.get(symbol)
            # 唯一写入入口已完整校验并转为不可变元组；复用时只检查周期边界。
            if history is None or history[-1][0] != d - self.KLINE_INTERVAL_MS:
                history = await self.gateway.call(
                    self.gateway.market_client.rest_api.kline_candlestick_data,
                    symbol=symbol,
                    interval="1d",
                    start_time=d - 29 * self.KLINE_INTERVAL_MS,
                    end_time=d - 1,
                    limit=29,
                )
                if (
                    isinstance(history, (list, tuple))
                    and 0 < len(history) < 29
                    and self._valid_history(history, d, count=len(history))
                ):
                    raise InsufficientKlineHistory(
                        f"{symbol}历史日K仅{len(history)}根，需要29根，本次跳过"
                    )
                if not self._valid_history(history, d):
                    raise ValueError("历史日K数量/日期/数值无效")
                history = tuple(tuple(row[:6]) for row in history)
                self.cache.kline_history[symbol] = history
            current = self._latest_bar(symbol)
            if current is None or current[0] != d:
                self.logger.warning(f"历史请求期间WS失效或跨日 symbol={symbol}")
                return MarketBars()
            return MarketBars.from_binance((*history, current))
        except InsufficientKlineHistory:
            raise
        except Exception:
            self.logger.exception(f"{symbol}获取历史K线失败")
            return MarketBars()

    @classmethod
    def _valid_history(cls, rows, d, count=29):
        if not isinstance(rows, (list, tuple)) or len(rows) != count:
            return False
        for i, row in enumerate(rows):
            if (
                not isinstance(row, (list, tuple))
                or len(row) < 6
                or type(row[0]) is not int
            ):
                return False
            if row[0] != d - (
                count - i
            ) * cls.KLINE_INTERVAL_MS or not cls._valid_ohlcv(row):
                return False
        return True

    @staticmethod
    def _bar_timestamp_ms(item):
        """统一成毫秒时间戳；缺失返回None"""
        raw = item.get("timestamp")
        if raw is None:
            return None
        timestamp_ms = int(float(raw))
        if timestamp_ms < 10**12:  # 容错：秒级时间戳换算成毫秒
            timestamp_ms *= 1000
        return timestamp_ms

    @classmethod
    def _is_current_5m_bar(cls, item) -> bool:
        """缓存那根是否已是当前5m边界那根：对上就不用再拉（时间戳是收盘时刻）"""
        timestamp_ms = cls._bar_timestamp_ms(item)
        if timestamp_ms is None:
            return False
        period_ms = cls.FIVE_MIN_PERIOD_MS
        return timestamp_ms == int(time.time() * 1000) // period_ms * period_ms

    @classmethod
    def _is_newer_bar(cls, fetched, cached) -> bool:
        """拉回来的比缓存里那根新才值得换；没缓存时任何带时间戳的都算新"""
        fetched_ms = cls._bar_timestamp_ms(fetched)
        if fetched_ms is None:
            return False
        if not cached:
            return True
        cached_ms = cls._bar_timestamp_ms(cached[-1])
        return cached_ms is None or fetched_ms > cached_ms

    async def oi_5m(self, symbol):
        return await self._get_5m_data(
            symbol,
            self.cache.oi_5m,
            self.gateway.market_client.rest_api.open_interest_statistics,
            "持仓量",
        )

    async def _get_5m_data(self, symbol, cache, api_method, data_name):
        """缓存已是当期那根就不拉；否则拉回来比时间戳，更新才换缓存，最后都用缓存。"""
        cached = cache.get(symbol)
        endpoint = "oi" if cache is self.cache.oi_5m else "lsr"
        if cached and self._is_current_5m_bar(cached[-1]):
            self.gateway.diagnostics.note("cache", endpoint, "5m", "hit")
            return cached
        self.gateway.diagnostics.note(
            "cache", endpoint, "5m", "lag_refetch" if cached else "miss"
        )

        try:
            data = await self.gateway.call(
                api_method,
                symbol=symbol,
                period="5m",
                limit=1,
            )
        except Exception as e:
            self.logger.error(
                f"{symbol}获取5m{data_name}数据失败: {type(e).__name__}: {e!r}"
            )
            return cache.get(symbol)

        if data and self._is_newer_bar(data[-1], cached):
            cache[symbol] = data
        self.gateway.diagnostics.note(
            "cache_result",
            endpoint,
            "5m",
            "updated" if cache.get(symbol) is not cached else "unchanged",
        )
        return cache.get(symbol)

    @staticmethod
    def _utc_history_boundary_ms(dtn: datetime, period: str) -> int:
        """返回历史序列当前可用的UTC周期边界，仅支持1d和1h。"""
        utc_boundary = datetime.datetime.fromtimestamp(
            dtn.timestamp(), datetime.timezone.utc
        ).replace(minute=0, second=0, microsecond=0)
        if period == "1d":
            utc_boundary = utc_boundary.replace(hour=0)
        elif period != "1h":
            raise ValueError(f"不支持的历史数据周期: {period}")
        return int(utc_boundary.timestamp() * 1000)

    async def oi_history(self, symbol: str, dtn: datetime, period: str):
        """按1d/1h周期隔离缓存历史OI，新的完整序列才替换对应缓存。"""
        available_before_ts = self._utc_history_boundary_ms(dtn, period)
        cache_key = (symbol, period)
        cached = self.cache.oi_history.get(cache_key)
        if cached and self._bar_timestamp_ms(cached[-1]) == available_before_ts:
            self.gateway.diagnostics.note("cache", "oi", period, "hit")
            return cached
        self.gateway.diagnostics.note(
            "cache", "oi", period, "lag_refetch" if cached else "miss"
        )

        try:
            oi_history = await self.gateway.call(
                self.gateway.market_client.rest_api.open_interest_statistics,
                symbol=symbol,
                period=period,
                limit=self.OI_QUERY_LIMIT,
            )
        except Exception as e:
            self.logger.error(
                f"{symbol}获取{period}持仓量数据失败: {type(e).__name__}: {e!r}"
            )
            return self.cache.oi_history.get(cache_key)

        available_oi = [
            item
            for item in (oi_history or [])
            if item.get("timestamp") is not None
            and int(item["timestamp"]) <= available_before_ts
        ]
        if len(available_oi) >= self.OI_QUERY_LIMIT and self._is_newer_bar(
            available_oi[-1], cached
        ):
            self.cache.oi_history[cache_key] = available_oi
        self.gateway.diagnostics.note(
            "cache_result",
            "oi",
            period,
            "updated"
            if self.cache.oi_history.get(cache_key) is not cached
            else "unchanged",
        )
        return self.cache.oi_history.get(cache_key)

    async def bd_oi_windows(self, symbol, dtn: datetime):
        """返回BD入池实时窗口和开仓使用的已完成日线OI窗口。"""
        oi_1d = await self.oi_history(symbol, dtn, "1d")
        if not oi_1d or len(oi_1d) < self.OI_QUERY_LIMIT:
            return None, None

        oi_5m = await self.oi_5m(symbol)
        if not oi_5m:
            return None, None
        latest_oi_5m = oi_5m[-1]

        completed_oi = [float(item["sumOpenInterest"]) for item in oi_1d]
        realtime_oi = completed_oi + [float(latest_oi_5m["sumOpenInterest"])]
        return realtime_oi, completed_oi

    async def lsr_1h(self, symbol: str, dtn: datetime):
        """缓存BZ做多所需的1h多空人数比，末根对上UTC小时边界就不拉。"""
        available_before_ts = self._utc_history_boundary_ms(dtn, "1h")

        cached = self.cache.lsr_1h.get(symbol)
        if cached and self._bar_timestamp_ms(cached[-1]) == available_before_ts:
            self.gateway.diagnostics.note("cache", "lsr", "1h", "hit")
            return cached
        self.gateway.diagnostics.note(
            "cache", "lsr", "1h", "lag_refetch" if cached else "miss"
        )

        try:
            data = await self.gateway.call(
                self.gateway.market_client.rest_api.long_short_ratio,
                symbol=symbol,
                period="1h",
                limit=self.LONG_SHORT_RATIO_LIMIT,
            )
        except Exception as e:
            self.logger.error(
                f"{symbol}获取1h多空比数据失败: {type(e).__name__}: {e!r}"
            )
            return self.cache.lsr_1h.get(symbol)

        available_lsr = [
            item
            for item in (data or [])
            if item.get("timestamp") is not None
            and int(item["timestamp"]) <= available_before_ts
        ]
        if len(available_lsr) >= self.LONG_SHORT_RATIO_LIMIT and self._is_newer_bar(
            available_lsr[-1], cached
        ):
            self.cache.lsr_1h[symbol] = available_lsr
        self.gateway.diagnostics.note(
            "cache_result",
            "lsr",
            "1h",
            "updated" if self.cache.lsr_1h.get(symbol) is not cached else "unchanged",
        )
        return self.cache.lsr_1h.get(symbol)

    async def lsr_5m(self, symbol: str):
        return await self._get_5m_data(
            symbol,
            self.cache.lsr_5m,
            self.gateway.market_client.rest_api.long_short_ratio,
            "多空比",
        )

    async def basis_rate(self, symbol: str) -> float:
        """
        获取指定交易对的基差率（带5分钟缓存）

        基差率 = (期货价格 - 指数价格) / 指数价格
        - 基差率 > 0：期货升水（期货价格高于现货）
        - 基差率 < 0：期货贴水（期货价格低于现货）

        参数:
            symbol: 交易对符号，如'BTCUSDT'
        返回:
            基差率（浮点数），获取失败返回0
        """
        try:
            # 检查缓存是否有效
            current_time = time.time()
            cache_entry = self.cache.basis_rate.get(symbol)
            if (
                cache_entry
                and (current_time - cache_entry["timestamp"])
                < self.BASIS_RATE_CACHE_TTL
            ):
                return cache_entry["data"]

            # 获取指数价格和标记价格（统一受全局并发闸门约束）
            premium_index = await self.gateway.call(
                self.gateway.market_client.rest_api.mark_price,
                symbol=symbol,
            )
            if not premium_index:
                return 0.0

            index_price = float(premium_index.get("indexPrice", 0))
            mark_price = float(premium_index.get("markPrice", 0))

            if index_price <= 0:
                return 0.0

            # 基差率 = (期货价格 - 指数价格) / 指数价格
            basis_rate = (mark_price - index_price) / index_price

            # 更新缓存
            self.cache.basis_rate[symbol] = {
                "data": basis_rate,
                "timestamp": current_time,
            }

            return basis_rate
        except Exception as e:
            self.logger.error(f"{symbol} 获取基差率失败: {str(e)}")
            return 0.0


class AshareMarketData(MarketData):
    """腾讯行情、东方财富涨停池、新浪交易日历；只使用现有HTTP依赖。"""

    TRADING_DAYS_LOOKBACK = 20
    MAX_QUOTE_AGE = datetime.timedelta(minutes=5)
    HEADERS = MappingProxyType(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.qq.com/",
        }
    )

    def __init__(self, http, logger):
        self.http, self.logger = http, logger
        self.cache = AshareCache()

    async def _request(self, url, *, params=None):
        for attempt in range(2):
            try:
                session = await self.http.get()
                async with session.get(
                    url, params=params, headers=self.HEADERS
                ) as response:
                    response.raise_for_status()
                    text = (await response.text()).strip()
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    wrapper = re.fullmatch(
                        r"[A-Za-z_$][\w.$]*\s*\((.*)\)\s*;?", text, re.DOTALL
                    )
                    if wrapper is None:
                        raise ValueError("行情响应不是JSON/JSONP")
                    payload = json.loads(wrapper.group(1))
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                if attempt:
                    raise
                await asyncio.sleep(0.25)
            else:
                if not isinstance(payload, dict):
                    raise ValueError("行情响应结构无效")
                return payload

    @staticmethod
    def _symbol(code):
        if not isinstance(code, str) or re.fullmatch(r"\d{6}", code) is None:
            raise ValueError("股票代码格式无效")
        return ("sh" if code.startswith("6") else "sz") + code

    @staticmethod
    def _ohlcv(values, *, realtime=False):
        bar = dict(
            zip(
                ("open", "high", "low", "close", "volume"),
                map(float, values),
                strict=True,
            )
        )
        if not all(math.isfinite(value) for value in bar.values()):
            raise ValueError("OHLCV含非有限数值")
        if (
            any(bar[key] <= 0 for key in ("open", "high", "low", "close"))
            or bar["volume"] < 0
        ):
            raise ValueError("OHLCV价格/成交量无效")
        if (
            not bar["low"]
            <= min(bar["open"], bar["close"])
            <= max(bar["open"], bar["close"])
            <= bar["high"]
        ):
            raise ValueError("OHLC价格范围不一致")
        if realtime and bar["volume"] == 0:
            raise ValueError("当日没有成交，暂不使用行情")
        # 腾讯日线和行情快照成交量均为手，保持策略单位。
        return bar

    async def is_trading_day(self, now):
        dates = await self.trading_calendar(now)
        return dates is not None and bool(
            (dates == pd.Timestamp(market_time(now).date())).any()
        )

    async def context(self, now):
        if not (9, 30) <= (now.hour, now.minute) < (15, 0):
            return None
        today = now.strftime("%Y-%m-%d")
        if not self.cache.trading_dates or self.cache.trading_dates[0] != today:
            self.cache.trading_dates = await self.recent_dates(now)
            self.cache.history.clear()
            if self.cache.trading_dates and self.cache.trading_dates[0] != today:
                return None
        if not self.cache.trading_dates:
            return MarketContext(
                now,
                (
                    now - datetime.timedelta(days=self.TRADING_DAYS_LOOKBACK * 3)
                ).strftime("%Y-%m-%d"),
                today,
                True,
            )
        return MarketContext(
            now, self.cache.trading_dates[-1], self.cache.trading_dates[0]
        )

    async def limit_up_pool(self, now):
        if not await self.is_trading_day(now):
            return (), None
        dates = await self.recent_dates(now)
        if not dates or dates[0] != now.strftime("%Y-%m-%d"):
            return (), None
        payload = await self._request(
            "https://push2ex.eastmoney.com/getTopicZTPool",
            params={
                "ut": "7eea3edcaed734bea9cbfc24409ed989",
                "dpt": "wz.ztzt",
                "Pageindex": 0,
                "pagesize": 10000,
                "sort": "fbt:asc",
                "date": now.strftime("%Y%m%d"),
            },
        )
        if payload.get("rc") != 0:
            raise ValueError("东方财富涨停池响应失败")
        data = payload.get("data")
        if data is None:
            return (), None
        rows = data["pool"]
        if not isinstance(rows, list):
            raise ValueError("东方财富涨停池结构无效")
        total = data.get("tc")
        if type(total) is not int or total < 0 or total != len(rows):
            raise ValueError("东方财富涨停池总数缺失或与实际数量不一致")
        if not rows:
            return (), None
        if len(rows) >= 10000:
            raise ValueError("涨停池达到请求上限，不能确认完整性")
        selected = {}
        for row in rows:
            code, name = row["c"], row["n"]
            self._symbol(code)
            if code in selected or not isinstance(name, str) or not name:
                raise ValueError("涨停池名称缺失或代码重复")
            selected[code] = name
        return tuple(selected.items()), MarketContext(now, dates[-1], dates[0])

    def invalidate_history(self):
        self.cache.trading_dates.clear()
        self.cache.history.clear()

    async def daily_bar(self, code, *, now=None):
        now = market_time(now)
        try:
            payload = await self._quote_payload(code)
            return self._current_bar(code, payload, now)
        except Exception as error:
            self.logger.warning(
                f"{code} 获取腾讯当日行情失败:{type(error).__name__}: {error}"
            )
            return None

    def _current_bar(self, code, payload, now):
        quote = payload["qt"][self._symbol(code)]
        if quote[2] != code:
            raise ValueError("腾讯行情股票代码不匹配")
        quote_time = datetime.datetime.strptime(quote[30], "%Y%m%d%H%M%S").replace(
            tzinfo=MARKET_TIMEZONE
        )
        if quote_time.date() != now.date() or quote_time > now + datetime.timedelta(
            minutes=1
        ):
            raise ValueError("腾讯行情日期/时间不匹配")
        expected = now
        if (11, 30) <= (now.hour, now.minute) < (13, 0):
            expected = now.replace(hour=11, minute=30, second=0, microsecond=0)
        elif now.hour >= 15:
            expected = now.replace(hour=15, minute=0, second=0, microsecond=0)
        if expected - quote_time > self.MAX_QUOTE_AGE:
            raise ValueError("腾讯当日行情已过期")
        return self._ohlcv((quote[i] for i in (5, 33, 34, 3, 6)), realtime=True)

    async def _quote_payload(self, code):
        symbol = self._symbol(code)
        payload = await self._request(
            "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
            params={"param": f"{symbol},day,,,30,qfq"},
        )
        if payload.get("code") != 0:
            raise ValueError("腾讯行情响应失败")
        return payload["data"][symbol]

    async def trading_calendar(self, today=None):
        anchor = market_time(today).date()
        loaded_on = market_time().date()
        if (
            self.cache.calendar is not None
            and self.cache.calendar_loaded_on == loaded_on
        ):
            cached = self.cache.calendar
            return (
                cached
                if cached.iloc[0] <= pd.Timestamp(anchor) <= cached.iloc[-1]
                else None
            )
        try:
            url = "https://finance.sina.com.cn/realstock/company/klc_td_sh.txt"
            for attempt in range(2):
                try:
                    session = await self.http.get()
                    async with session.get(url, headers=self.HEADERS) as response:
                        response.raise_for_status()
                        dates = self._decode_calendar(await response.text())
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                    if attempt:
                        raise
                    await asyncio.sleep(0.25)
            dates = pd.Series(
                sorted(set([*dates, pd.Timestamp("1992-05-04")])),
                dtype="datetime64[ns]",
            )
            if (
                len(dates) < self.TRADING_DAYS_LOOKBACK
                or not dates.iloc[0] <= pd.Timestamp(anchor) <= dates.iloc[-1]
            ):
                raise ValueError("新浪交易日历不完整")
            self.cache.calendar = dates
            self.cache.calendar_loaded_on = loaded_on
        except Exception as error:
            self.logger.error(f"获取新浪交易日历失败:{type(error).__name__}: {error}")
        cached = self.cache.calendar
        return (
            cached
            if cached is not None
            and cached.iloc[0] <= pd.Timestamp(anchor) <= cached.iloc[-1]
            else None
        )

    @staticmethod
    def _decode_calendar(payload):
        # 新浪KLC交易日历：6位字符按低位优先读取，日期使用工作日游程编码。
        match = re.fullmatch(
            r'\s*var datelist=("[A-Za-z0-9+/]+");\s*var KLC_TD_SH=datelist;\s*', payload
        )
        if match is None:
            raise ValueError("新浪交易日历响应格式无效")
        encoded = json.loads(match.group(1))
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        digits = [alphabet.index(c) for c in encoded]
        bit = 0

        def read(width):
            nonlocal bit
            if width < 0 or width > 30 or bit + width > len(digits) * 6:
                raise ValueError("新浪交易日历位流截断或长度无效")
            value = 0
            for shift in range(width):
                value |= ((digits[bit // 6] >> (bit % 6)) & 1) << shift
                bit += 1
            return value

        if read(12) != 139 or (63 ^ read(6)) > 1:
            raise ValueError("新浪交易日历编码版本无效")
        day = read(18) - 1
        end = read(18)
        length = 0
        remaining = -1
        dates = []
        while day < end:
            day += 1
            if day % 7 in (3, 4):
                day += 5 - day % 7
            date = pd.Timestamp("1970-01-01") + pd.Timedelta(days=7657 + day)
            if remaining <= 0:
                if read(1):
                    sign = 2 * read(1) - 1
                    count = 1
                    while read(1):
                        count += 1
                    length += sign * count
                remaining = read(3 * length) + 1
                if not dates:
                    dates.append(date)
                    remaining -= 1
            else:
                dates.append(date)
            remaining -= 1
        return dates

    async def recent_dates(self, today=None, days=None):
        today = market_time(today)
        days = self.TRADING_DAYS_LOOKBACK if days is None else days
        dates = await self.trading_calendar(today)
        if dates is None:
            return []
        recent = (
            dates[dates <= pd.Timestamp(today.date())]
            .sort_values(ascending=False)
            .iloc[:days]
        )
        return [item.strftime("%Y-%m-%d") for item in recent]

    async def following_dates(self, today, days=2, *, calendar=None):
        today = market_time(today)
        if (
            calendar is None
            or calendar.empty
            or pd.Timestamp(today.date()) < calendar.iloc[0]
        ):
            calendar = await self.trading_calendar(today)
        if calendar is None:
            return []
        following = calendar[calendar > pd.Timestamp(today.date())].iloc[:days]
        return [item.strftime("%Y-%m-%d") for item in following]

    async def reopen_timestamp(self, today, *, calendar=None):
        following = await self.following_dates(today, days=2, calendar=calendar)
        if len(following) < 2:
            return None
        return (
            datetime.datetime.strptime(following[1], "%Y-%m-%d")
            .replace(tzinfo=MARKET_TIMEZONE)
            .timestamp()
        )

    async def fetch_bars(self, code, context):
        end_date = pd.Timestamp(
            context.end_date or market_time(context.now).date()
        ).normalize()
        start_date = pd.Timestamp(context.start_date or "1970-01-01").normalize()
        if end_date.date() != market_time(context.now).date():
            return MarketBars()
        try:
            # 单次腾讯响应同时含历史线和当日行情，避免每币重复取数。
            payload = await self._quote_payload(code)
            current_bar = self._current_bar(code, payload, market_time(context.now))
            rows = payload.get("qfqday")
            if not isinstance(rows, list) or not rows:
                raise ValueError("腾讯前复权历史日线为空")
            records = []
            for row in rows:
                date = pd.Timestamp(datetime.datetime.strptime(row[0], "%Y-%m-%d"))
                if start_date <= date < end_date:
                    records.append(
                        {"date": date, **self._ohlcv((row[i] for i in (1, 3, 4, 2, 5)))}
                    )
            if not records:
                raise ValueError("指定日期范围无历史日线")
            history = pd.DataFrame(records)
            if (
                history["date"].duplicated().any()
                or not history["date"].is_monotonic_increasing
            ):
                raise ValueError("腾讯历史日线乱序或重复")
            return MarketBars.from_frame(
                pd.concat(
                    [history, pd.DataFrame([{"date": end_date, **current_bar}])],
                    ignore_index=True,
                )
            )
        except Exception as error:
            self.logger.warning(
                f"{code} 获取腾讯K线失败:{type(error).__name__}: {error}"
            )
            return MarketBars()


class TradeExecutor(ABC):
    REQUIRES_ORDER_JOURNAL = False

    @abstractmethod
    async def prepare(self, intent: TradeIntent) -> PreparedOrder | None: ...

    @abstractmethod
    async def submit(self, order: PreparedOrder) -> ExecutionResult: ...

    @abstractmethod
    async def reconcile(self, order: PreparedOrder) -> ExecutionResult: ...

    async def complete_result(
        self, order: PreparedOrder, result: ExecutionResult
    ) -> ExecutionResult:
        return result


class BinanceExecutor(TradeExecutor):
    REQUIRES_ORDER_JOURNAL = True
    DEFINITE_REJECTION_CODES = frozenset(
        {
            -1013,
            -1015,
            -1021,
            -1022,
            -2010,
            -2014,
            -2015,
            -2018,
            -2019,
            -2020,
            -2021,
            -2022,
            -2024,
            -2025,
            -2026,
            -2027,
            -2028,
        }
    )
    RISK_PER_TRADE = 0.1
    TARGET_PROFIT_RATIO = 0.1
    MAX_POSITION_RATIO = 0.1
    MAINTENANCE_MARGIN_RATE = 0.004
    MIN_NOTIONAL = 10
    CLOSE_RETRY_DELAY = 3

    def __init__(self, gateway, market_data, logger, *, leverage=5, health4open=70):
        self.gateway, self.market_data, self.logger = gateway, market_data, logger
        self.leverage, self.health4open = leverage, health4open

    def close_quantity(self, symbol, amount, ratio):
        precision = self.market_data.precision(symbol)
        if precision is None:
            if ratio == 1:
                return amount
            self.logger.warning(f"{symbol} 缺少精度，暂缓部分平仓")
            return None
        return float(
            Decimal(str(amount * ratio)).quantize(precision, rounding=ROUND_DOWN)
        )

    async def prepare(self, intent):
        try:
            if intent.kind == ActionKind.CLOSE:
                amount_price = await self.gateway.position_amount(
                    intent.symbol, intent.position.position_side.value
                )
                if amount_price is None:
                    self.logger.warning(f"{intent.symbol}查询持仓数量失败")
                    return None
                amount, entry = amount_price
                if not amount:
                    return PreparedOrder(
                        intent, 0, intent.price, entry, no_position=True
                    )
                ratio = (
                    1
                    if amount * intent.ratio * intent.price < self.MIN_NOTIONAL
                    else intent.ratio
                )
                quantity = self.close_quantity(intent.symbol, amount, ratio)
                if quantity is None or quantity <= 0:
                    return None
                return PreparedOrder(
                    intent,
                    quantity,
                    intent.price,
                    entry if entry > 0 else intent.position.entry_price,
                    ratio,
                )
            position = intent.position
            precision = self.market_data.precision(intent.symbol)
            if (
                precision is None
                or position.stop_loss <= 0
                or position.take_profit <= 0
            ):
                return None
            rows = await self.gateway.positions()
            symbol_positions = [p for p in rows if p.get("symbol") == intent.symbol]
            if intent.kind == ActionKind.OPEN and symbol_positions:
                self.logger.error(
                    f"{intent.symbol} 账户已有持仓但本地无对应策略记录，拒绝重复开仓；需要恢复策略状态"
                )
                return None
            if intent.kind == ActionKind.ADD and not any(
                p.get("positionSide") == position.position_side.value
                for p in symbol_positions
            ):
                self.logger.warning(
                    f"{intent.symbol} 已确认对应方向无持仓，取消加仓并清理本地旧记录"
                )
                return PreparedOrder(
                    intent, 0, intent.price, position.entry_price, no_position=True
                )
            account, mark = await asyncio.gather(
                self.gateway.account(),
                self.gateway.call(
                    self.gateway.market_client.rest_api.mark_price, symbol=intent.symbol
                ),
            )
            if account["available"] <= 0 or not mark:
                return None
            price = float(mark["markPrice"])
            sl_gap, tp_gap = (
                abs(price - position.stop_loss),
                abs(position.take_profit - price),
            )
            if sl_gap <= 0 or tp_gap <= 0:
                return None
            amount_raw = (
                account["equity"] * self.RISK_PER_TRADE / sl_gap
                + account["equity"] * self.TARGET_PROFIT_RATIO / tp_gap
            ) / 2
            max_amount = account["available"] * self.MAX_POSITION_RATIO / price
            if amount_raw > max_amount:
                amount_raw = max_amount
            quantity = float(
                Decimal(str(amount_raw)).quantize(precision, rounding=ROUND_DOWN)
            )
            notional = quantity * price
            if (
                notional < self.MIN_NOTIONAL
                or notional / self.leverage > account["available"]
            ):
                return None
            maintenance = sum(
                float(
                    p.get("maintMargin", 0)
                    or abs(float(p.get("notional", 0) or 0))
                    * self.MAINTENANCE_MARGIN_RATE
                )
                for p in rows
            )
            if not account["maintenance"]:
                maintenance += notional * self.MAINTENANCE_MARGIN_RATE
            equity = account["equity"]
            health = (
                100
                if maintenance == 0 and equity >= 0
                else 0
                if equity <= 0
                else round(min(100, max(0, (1 - maintenance / equity) * 100)))
            )
            if health < self.health4open:
                return None
            leverage = await self.gateway.call(
                self.gateway.papi_client.rest_api.change_um_initial_leverage,
                symbol=intent.symbol,
                leverage=self.leverage,
                recv_window=60000,
            )
            return PreparedOrder(
                intent,
                quantity,
                price,
                position.entry_price,
                leverage=leverage.get("leverage", self.leverage),
                account_equity=equity,
                notional=notional,
            )
        except Exception:
            self.logger.exception(f"{intent.symbol}交易准备失败")
            return None

    @staticmethod
    def _response_result(order, response):
        status = response.get("status")
        executed = float(response.get("executedQty", 0) or 0)
        if status == "FILLED" or (
            status in ("CANCELED", "EXPIRED", "EXPIRED_IN_MATCH") and executed > 0
        ):
            return ExecutionResult(
                ExecutionStatus.FILLED,
                executed or order.quantity,
                order.price,
                order.entry_price,
                order.ratio,
                response.get("origQty", order.quantity),
            )
        if status in ("REJECTED", "CANCELED", "EXPIRED", "EXPIRED_IN_MATCH"):
            return ExecutionResult(ExecutionStatus.FAILED)
        return ExecutionResult(ExecutionStatus.UNKNOWN)

    async def submit(self, order):
        intent = order.intent
        if order.no_position:
            return ExecutionResult(ExecutionStatus.NO_POSITION, remaining_quantity=0)
        side = (
            intent.position.close_side
            if intent.kind == ActionKind.CLOSE
            else intent.position.open_side
        )
        try:
            response = await self.gateway.submit(
                intent.symbol,
                side.value,
                intent.position.position_side.value,
                order.quantity,
                order.client_order_id,
            )
            result = self._response_result(order, response)
            return (
                await self.reconcile(order)
                if result.status == ExecutionStatus.UNKNOWN
                else result
            )
        except asyncio.CancelledError:
            self.logger.warning(f"{intent.symbol}提交已取消，先查询订单确认结果")
            return await self.reconcile(order)
        except Exception as error:
            if self._definite_rejection(error):
                self.logger.warning(
                    f"{intent.symbol}订单明确拒绝 code:{getattr(error, 'status_code', None)}，持仓状态保持"
                )
                return ExecutionResult(ExecutionStatus.FAILED)
            self.logger.warning(
                f"{intent.symbol}订单结果待确认({type(error).__name__})，禁止重新提交"
            )
            return await self.reconcile(order)

    @classmethod
    def _definite_rejection(cls, error):
        if isinstance(
            error,
            (
                RequiredError,
                UnauthorizedError,
                ForbiddenError,
                TooManyRequestsError,
                RateLimitBanError,
            ),
        ):
            return True
        code = getattr(error, "status_code", None)
        return (
            isinstance(error, BadRequestError)
            and isinstance(code, int)
            and (code in cls.DEFINITE_REJECTION_CODES or -1199 <= code <= -1100)
        )

    async def reconcile(self, order):
        try:
            response = await self.gateway.query_order(
                order.intent.symbol, order.client_order_id
            )
            return self._response_result(order, response)
        except Exception:
            self.logger.warning(f"{order.intent.symbol}订单查询未完成，保留待确认状态")
            return ExecutionResult(ExecutionStatus.UNKNOWN)

    async def complete_result(self, order, result):
        if order.intent.kind == ActionKind.OPEN or not result.filled:
            return result
        latest = await self.gateway.position_amount(
            order.intent.symbol, order.intent.position.position_side.value
        )
        if latest is None:
            self.logger.warning(
                f"{order.intent.symbol}成交后持仓查询失败，保留已确认的状态"
            )
            return result
        return replace(
            result,
            remaining_quantity=latest[0],
            average_price=latest[1] if latest[1] > 0 else None,
        )


class AshareExecutor(TradeExecutor):
    SHARES_PER_ORDER = 100

    async def reconcile(self, order):
        # 旧版快照可能遗留虚拟 pending；没有外部订单，丢弃该动作后按最新行情重评。
        return ExecutionResult(ExecutionStatus.FAILED)

    async def prepare(self, intent):
        quantity = (
            self.SHARES_PER_ORDER * (1 + intent.position.dca_count)
            if intent.kind == ActionKind.CLOSE
            else self.SHARES_PER_ORDER
        )
        entry = (
            intent.position.entry_price
            if intent.position.entry_price > 0
            else intent.price
        )
        return PreparedOrder(intent, quantity, intent.price, entry)

    async def submit(self, order):
        intent = order.intent
        average = None
        if intent.kind == ActionKind.ADD:
            shares = self.SHARES_PER_ORDER * (1 + intent.position.dca_count)
            average = (
                (
                    intent.position.entry_price * shares
                    + order.price * self.SHARES_PER_ORDER
                )
                / (shares + self.SHARES_PER_ORDER)
                if intent.position.entry_price > 0
                else order.price
            )
        return ExecutionResult(
            ExecutionStatus.FILLED,
            order.quantity,
            order.price,
            order.entry_price,
            1,
            order.quantity,
            average,
            0 if intent.kind == ActionKind.CLOSE else None,
        )


class MarketStrategy(ABC):
    MARKET: str
    SIGNAL_CHANNEL: str
    SIGNAL_BEFORE_ORDER = False
    CYCLE_HOURS = "*"
    WEEKDAYS = "*"
    DAILY_HOUR = 8
    DAILY_MINUTE = 0
    SHARES_PER_ORDER = 100

    def __init__(self, data):
        self.data = data

    @abstractmethod
    async def prepare_cycle(
        self, now, state: StateSnapshot
    ) -> tuple[MarketContext | None, StatePatch]: ...

    @abstractmethod
    def scan_candidates(self, state: StateSnapshot) -> tuple[str, ...]: ...

    async def prepare_instrument(self, symbol, state, context):
        return Decision()

    @abstractmethod
    async def evaluate_signal(
        self, symbol, bars: MarketBars, state: StateSnapshot, context
    ) -> Decision: ...

    @abstractmethod
    async def manage_position(
        self, symbol, bars: MarketBars, state: StateSnapshot, context
    ) -> Decision: ...

    async def after_instrument(self, symbol, bars, state, prior, context) -> StatePatch:
        return StatePatch()

    @abstractmethod
    async def refresh_universe(
        self, now, state: StateSnapshot
    ) -> tuple[bool, StatePatch]: ...

    async def daily_report_due(self, now):
        return True

    @abstractmethod
    async def build_daily_report(
        self, now, state: StateSnapshot
    ) -> TradeNotification: ...

    def observation_after_open(self, intent):
        return intent.observation

    @abstractmethod
    def observation_after_close(
        self, intent, state: StateSnapshot, now
    ) -> Observation | None: ...

    @abstractmethod
    def position_after_add(self, intent, result) -> Position: ...

    @abstractmethod
    def position_after_take_profit(self, intent, result) -> Position: ...

    @abstractmethod
    def signal_message(self, intent) -> str: ...

    @abstractmethod
    def trade_message(
        self, order, result, position: Position
    ) -> TradeNotification | None: ...

    def record_symbol(self, symbol, position):
        return symbol


class TradingEngine:
    CYCLE_TIMEOUT_SECONDS = 55
    DAILY_TIMEOUT_SECONDS = 180
    BATCH_WINDOW_MINUTES = 5
    BATCH_SLOT_COUNT = 5
    MAX_CONCURRENT_REQUESTS = 8

    def __init__(self, strategy, executor, state, notifications, records, http, logger):
        self.strategy, self.executor, self.state = strategy, executor, state
        self.notifications, self.records, self.http, self.logger = (
            notifications,
            records,
            http,
            logger,
        )
        self.cache = ScanCache()
        self._cycle_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
        self._committed_orders = set()

    def initialize(self):
        self.state.load()
        atexit.register(self.state.save)

    async def close(self):
        async with self._cycle_lock:
            await self.strategy.data.close()
            await self.http.close()

    async def run_market_cycle(self, now=None):
        async with self._cycle_lock:
            try:
                await asyncio.wait_for(
                    self._scan(
                        market_time(now), time.monotonic() + self.CYCLE_TIMEOUT_SECONDS
                    ),
                    self.CYCLE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                self.logger.info("扫描窗口到期，未完成标的保留待后续扫描")
            except Exception:
                self.logger.exception("扫描失败")
            finally:
                await self.records.flush_pending_records()

    async def _scan(self, now, deadline=None):
        context, patch = await self.strategy.prepare_cycle(now, self.state.snapshot())
        self.state.apply(patch)
        if context is None:
            return
        window = int(now.timestamp() // (self.BATCH_WINDOW_MINUTES * 60))
        slot = now.minute % self.BATCH_SLOT_COUNT
        if not context.exit_only and self.cache.window_id != window:
            self.cache.window_id = window
            self.cache.snapshot = tuple(
                sorted(self.strategy.scan_candidates(self.state.snapshot()))
            )
        candidates = (
            ()
            if context.exit_only
            else self.cache.snapshot[slot :: self.BATCH_SLOT_COUNT]
        )
        symbols = tuple(
            dict.fromkeys(
                (
                    *candidates,
                    *self.state.snapshot().positions,
                    *self.state.pending_symbols(),
                )
            )
        )
        deadline = (
            deadline
            if deadline is not None
            else time.monotonic() + self.CYCLE_TIMEOUT_SECONDS
        )
        try:
            symbols = await self.strategy.data.begin_slot(symbols, deadline)
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(self._process_slot_symbol(symbol, context))
                    for symbol in symbols
                ]
        finally:
            await self.strategy.data.end_slot()
        outcomes = Counter(
            t.result() if not t.cancelled() else ScanOutcome.FAILED for t in tasks
        )
        self.logger.info(
            f"扫描完成 window:{window} slot:{slot} 成功:{outcomes[ScanOutcome.PROCESSED]}/{len(symbols)} "
            f"跳过:{outcomes[ScanOutcome.SKIPPED]} 无行情:{outcomes[ScanOutcome.NO_DATA]} "
            f"待确认:{outcomes[ScanOutcome.PENDING]} 失败:{outcomes[ScanOutcome.FAILED]}"
        )

    async def _process_slot_symbol(self, symbol, context):
        outcome = await self.process_instrument(symbol, context)
        # 已判断无信号/规则跳过也算本次完成；订单待确认走原pending链，不能重跑信号。
        if outcome in (ScanOutcome.PROCESSED, ScanOutcome.SKIPPED, ScanOutcome.PENDING):
            self.strategy.data.complete_slot_symbol(symbol)
        return outcome

    async def process_instrument(self, symbol, context):
        diagnostic_scope = _BINANCE_REQUEST_SCOPE.set("other")
        try:
            pending = self.state.pending_order(symbol)
            if pending is not None:
                _BINANCE_REQUEST_SCOPE.set("pending")
                async with self._semaphore:
                    result = await self.executor.reconcile(pending)
                    await self._finish_execution(pending, result, context, notify=True)
                return (
                    ScanOutcome.PENDING
                    if result.status == ExecutionStatus.UNKNOWN
                    else ScanOutcome.PROCESSED
                )
            prior = self.state.snapshot()
            _BINANCE_REQUEST_SCOPE.set(
                "held"
                if symbol in prior.positions
                else "observed"
                if symbol in prior.observations
                else "unobserved"
            )
            decision = await self.strategy.prepare_instrument(symbol, prior, context)
            self.state.apply(decision.patch)
            if not decision.proceed:
                return ScanOutcome.SKIPPED
            # 等待WS不占用交易/历史请求并发名额，避免慢标的挡住已就绪标的。
            if not await self.strategy.data.wait_ready(symbol):
                return ScanOutcome.NO_DATA
            async with self._semaphore:
                bars = await self.strategy.data.fetch_bars(symbol, context)
                if not bars:
                    return ScanOutcome.NO_DATA
                state = self.state.snapshot()
                decision = await (
                    self.strategy.manage_position(symbol, bars, state, context)
                    if symbol in state.positions
                    else self.strategy.evaluate_signal(symbol, bars, state, context)
                )
                self.state.apply(decision.patch)
                if decision.intent is not None:
                    if not await self.execute_action(decision.intent, context):
                        if self.state.pending_order(symbol) is not None:
                            return ScanOutcome.PENDING
                        # 原开仓失败会终止本轮；已有持仓的失败加仓/平仓仍执行观察后处理。
                        if decision.intent.kind == ActionKind.OPEN:
                            return ScanOutcome.PROCESSED
                self.state.apply(
                    await self.strategy.after_instrument(
                        symbol, bars, self.state.snapshot(), prior, context
                    )
                )
            return ScanOutcome.PROCESSED
        except InsufficientKlineHistory as error:
            self.logger.info(str(error))
            return ScanOutcome.SKIPPED
        except Exception:
            self.logger.exception(f"{symbol}处理失败")
            return ScanOutcome.FAILED
        finally:
            _BINANCE_REQUEST_SCOPE.reset(diagnostic_scope)

    def _remove_position(self, intent, context):
        observation = self.strategy.observation_after_close(
            intent, self.state.snapshot(), context.now
        )
        self.state.apply(
            StatePatch(
                positions=((intent.symbol, None),),
                observations=((intent.symbol, observation),),
            )
        )

    def _commit_fill(self, intent, result, execution_id=""):
        if intent.kind == ActionKind.OPEN:
            self.state.apply(
                StatePatch(
                    positions=((intent.symbol, intent.position),),
                    observations=(
                        (intent.symbol, self.strategy.observation_after_open(intent)),
                    ),
                )
            )
        elif intent.kind == ActionKind.ADD:
            self.state.apply(
                StatePatch(
                    positions=(
                        (
                            intent.symbol,
                            self.strategy.position_after_add(intent, result),
                        ),
                    )
                )
            )
        elif result.ratio < 1:
            self.state.apply(
                StatePatch(
                    positions=(
                        (
                            intent.symbol,
                            self.strategy.position_after_take_profit(intent, result),
                        ),
                    )
                )
            )
        if intent.kind == ActionKind.CLOSE:
            direction = 1 if intent.position.position_side == PositionSide.LONG else -1
            gain = (result.price - result.entry_price) * direction
            self.records.enqueue_close(
                source=self.strategy.MARKET,
                symbol=self.strategy.record_symbol(intent.symbol, intent.position),
                position_side=intent.position.position_side.value,
                entry_price=result.entry_price,
                close_price=result.price,
                close_amount=result.quantity,
                realized_pnl=gain * result.quantity,
                pnl_percent=gain / result.entry_price if result.entry_price else 0,
                close_ratio=result.ratio,
                strategy_tag=format_strategy_tags(intent.position.strategy),
                close_reason=intent.position.close_reason,
                long_short_ratio=intent.position.long_short_ratio,
                basis_rate=intent.position.basis_rate,
                execution_id=execution_id,
            )

    async def execute_action(self, intent, context):
        if self.state.pending_order(intent.symbol) is not None:
            return False
        order = await self.executor.prepare(intent)
        if order is None:
            return False
        if (
            intent.kind == ActionKind.OPEN
            and self.strategy.SIGNAL_BEFORE_ORDER
            and not order.no_position
        ):
            await self.notifications.send(
                self.strategy.signal_message(intent),
                channel=self.strategy.SIGNAL_CHANNEL,
            )
        if self.executor.REQUIRES_ORDER_JOURNAL and not order.no_position:
            order = replace(order, client_order_id="at-" + uuid4().hex)
            try:
                self.state.remember_order(order, durable=True)
            except Exception:
                self.logger.exception("真实订单恢复日志保存失败，取消提交")
                return False
        submitting = asyncio.create_task(self.executor.submit(order))
        try:
            result = await asyncio.shield(submitting)
        except asyncio.CancelledError:
            try:
                result = await submitting
                await self._finish_execution(order, result, context, notify=True)
            finally:
                raise
        return await self._finish_execution(order, result, context)

    async def _finish_execution(self, order, result, context, *, notify=True):
        intent = order.intent
        durable = (
            self.executor.REQUIRES_ORDER_JOURNAL
            and self.state.pending_order(intent.symbol) is not None
        )
        if result.status == ExecutionStatus.UNKNOWN:
            return False
        if result.status == ExecutionStatus.NO_POSITION:
            self._remove_position(intent, context)
            self.state.resolve_order(intent.symbol, durable=durable)
            return False
        if not result.filled:
            self.state.resolve_order(intent.symbol, durable=durable)
            return False
        # 已确认成交先提交；随后查询或通知取消不能撤销它。
        if (
            not order.client_order_id
            or order.client_order_id not in self._committed_orders
        ):
            self._commit_fill(intent, result, order.client_order_id)
            if order.client_order_id:
                self._committed_orders.add(order.client_order_id)
        if result.remaining_quantity == 0 and intent.kind == ActionKind.CLOSE:
            self._remove_position(intent, context)
        try:
            result = await self.executor.complete_result(order, result)
            if intent.kind == ActionKind.ADD and result.average_price is not None:
                self.state.apply(
                    StatePatch(
                        positions=(
                            (
                                intent.symbol,
                                self.strategy.position_after_add(intent, result),
                            ),
                        )
                    )
                )
            if intent.kind == ActionKind.CLOSE:
                if result.remaining_quantity == 0:
                    if intent.symbol in self.state.snapshot().positions:
                        self._remove_position(intent, context)
                elif result.ratio >= 1 and result.remaining_quantity is not None:
                    self.state.apply(
                        StatePatch(
                            positions=(
                                (
                                    intent.symbol,
                                    self.strategy.position_after_take_profit(
                                        intent, result
                                    ),
                                ),
                            )
                        )
                    )
        finally:
            # 成交后的补查取消也保留已确定状态；全量快照只在日报/退出保存。
            self.state.resolve_order(intent.symbol, durable=durable)
            self._committed_orders.discard(order.client_order_id)
        position = self.state.snapshot().positions.get(intent.symbol, intent.position)
        if not notify:
            return True
        if intent.kind == ActionKind.OPEN and not self.strategy.SIGNAL_BEFORE_ORDER:
            await self.notifications.send(
                self.strategy.signal_message(intent),
                channel=self.strategy.SIGNAL_CHANNEL,
            )
        message = self.strategy.trade_message(order, result, position)
        if message:
            await self.notifications.send(message, channel="pushplus")
        return True

    async def push_daily_report(self, now=None):
        async with self._cycle_lock:
            now = market_time(now)
            try:
                if not await self.strategy.daily_report_due(now):
                    return
                try:
                    try:
                        report = await asyncio.wait_for(
                            self.strategy.build_daily_report(
                                now, self.state.snapshot()
                            ),
                            self.DAILY_TIMEOUT_SECONDS,
                        )
                        await asyncio.wait_for(
                            self.notifications.send(report, channel="pushplus"),
                            self.http.timeout,
                        )
                    except Exception:
                        self.logger.exception("每日报告失败")
                    try:
                        changed, patch = await asyncio.wait_for(
                            self.strategy.refresh_universe(now, self.state.snapshot()),
                            self.DAILY_TIMEOUT_SECONDS,
                        )
                        self.state.apply(patch)
                        if changed:
                            self.cache.window_id = None
                    except Exception:
                        self.logger.exception("标的池刷新失败")
                finally:
                    try:
                        self.state.save()
                    except Exception:
                        self.logger.exception("保存失败，保留旧文件")
            finally:
                await self.records.flush_pending_records()


class AUTOBN(MarketStrategy):
    MARKET = "AUTOBN"
    SIGNAL_CHANNEL = "feishu"
    SIGNAL_BEFORE_ORDER = True
    ATR_PERIOD = 7
    TAKE_PROFIT_ATR_FACTOR = 3.0
    STOP_LOSS_ATR_FACTOR = 1.0
    ATR_HL2_CAP_RATIO = 0.1
    ATR_TRIGGER_CAP_RATIO = 0.05
    MIN_ATR_TRIGGER = 1e-8
    DCA_TP_ATR_RATIO = 0.5
    STOP_LOSS_DECAY_PER_MINUTE = 0.0001
    VOLUME_LOOKBACK_PERIOD = 10
    BZ_OBSERVATION_TIMEOUT_SECONDS = 86400
    BD_OBSERVATION_TIMEOUT_SECONDS = 7 * 86400
    REOPEN_COOLDOWN_SECONDS = 86400
    PARTIAL_CLOSE_RATIO = 0.7
    TRAILING_STOP_PROFIT_RATIO = 0.7
    BASIS_RATE_THRESHOLD = 0.02
    BZ_LONG_OI_DRAWDOWN_RATIO = 0.1
    BD_OI_DRAWDOWN_RATIO = 0.1
    CHEBYSHEV_EXTREME_THRESHOLD = 0.01
    SHORT_OI_CHEB_THRESHOLD = 0.05
    BD_VOLUME_RECENT_COUNT = 3
    BD_VOLUME_LOOKBACK_COUNT = 10
    BD_OI_LOOKBACK_COUNT = 10
    BD_OI_RECENT_COUNT = 3
    LONG_OI_CHEB_EXCLUDE_RECENT_COUNT = 6
    MIN_CHEB_SAMPLE_SIZE = 2
    TARGET_PROFIT_DIVISOR = 3.0
    OI_QUERY_LIMIT = 30
    OI_DELTA_LONG_RATIO_WEIGHT = 0.4

    async def prepare_cycle(self, now, state):
        await self.data.bootstrap(now)
        patch = self._positions_patch(self.data.positions_snapshot(now), state)
        return MarketContext(now), patch

    def _positions_patch(self, rows, state):
        if rows is None:
            return StatePatch()
        actual = {(row["symbol"], row["positionSide"]): row for row in rows}
        changes = []
        for symbol, held in state.positions.items():
            row = actual.get((symbol, held.position_side.value))
            if row is None:
                changes.append((symbol, None))
                continue
            entry = float(row.get("entryPrice", 0) or 0)
            if entry > 0 and entry != held.entry_price:
                changes.append((symbol, held.model_copy(update={"entry_price": entry})))
        pending_recovery = set()
        for symbol, side in actual:
            held = state.positions.get(symbol)
            if held is None or held.position_side.value != side:
                pending_recovery.add(symbol)
                self.data.require_position_recovery(symbol)
                self.data.logger.warning(
                    f"{symbol} {side} 缺少本地策略记录，进入N恢复流程"
                )
        for symbol in self.data.recovery_symbols():
            if symbol not in pending_recovery:
                self.data.resolve_position_recovery(symbol)
        return StatePatch(positions=tuple(changes))

    def scan_candidates(self, state):
        return tuple(
            dict.fromkeys(
                (
                    *self.data.universe(),
                    *self.data.recovery_symbols(),
                )
            )
        )

    async def prepare_instrument(self, symbol, state, context):
        if symbol not in state.positions and symbol in self.data.recovery_symbols():
            return Decision()
        obs = state.observations.get(symbol)
        if symbol not in state.positions and obs:
            if obs.is_reopen_cooldown_active(context.now.timestamp()):
                return Decision(proceed=False)
            duration = (
                self.BZ_OBSERVATION_TIMEOUT_SECONDS
                if StrategyTag.BZ in obs.strategy
                else self.BD_OBSERVATION_TIMEOUT_SECONDS
            )
            if context.now.timestamp() - obs.timestamp > duration:
                return Decision(StatePatch(observations=((symbol, None),)))
        return Decision()

    async def refresh_universe(self, now, state):
        refreshed = await self.data.refresh(now)
        patch = self._positions_patch(self.data.positions_snapshot(now), state)
        return refreshed, patch

    async def build_daily_report(self, now, state):
        rows = await self.data.account_positions(now)
        account = await self.data.gateway.account()
        positions = []
        for row in rows:
            held = state.positions.get(row["symbol"])
            notional = abs(float(row.get("notional", 0) or 0))
            if (
                held is None
                or row.get("positionSide") != held.position_side.value
                or notional == 0
            ):
                continue
            pnl = float(row.get("unRealizedProfit", 0) or 0)
            positions.append(
                DailyPosition(
                    name=row["symbol"],
                    strategy=format_strategy_tags(held.strategy),
                    direction=row.get("positionSide") or "LONG",
                    entry_price=float(row.get("entryPrice", 0)),
                    notional=notional,
                    unrealized_pnl=pnl,
                    profit_rate=pnl / notional,
                    take_profit=held.take_profit,
                    stop_loss=held.stop_loss,
                    open_date=held.date,
                    currency="USDT",
                )
            )
        return format_daily_positions_notification(
            self.MARKET, "账户余额", account["equity"], "USDT", positions
        )

    async def evaluate_signal(self, symbol, bars, state, context):
        if symbol in self.data.recovery_symbols():
            return await self._recover_unregistered_position(
                symbol, bars, state, context
            )
        obs = state.observations.get(symbol)
        if obs is None:
            return Decision()
        delta = bars.price - bars.closes[-2]
        is_long = delta > 0
        if (
            delta == 0
            or (StrategyTag.BZ if is_long else StrategyTag.BD) not in obs.strategy
        ):
            return Decision()
        side = PositionSide.LONG if is_long else PositionSide.SHORT
        passed, ratio, threshold = await self.check_side(
            symbol, side.value, context.now
        )
        if not passed:
            return Decision()
        atr = self.calculate_atr(bars)
        target, stop = self.calc_stop_profit_loss(bars.midprice, is_long, atr)
        if target == 0 and stop == 0:
            return Decision()
        basis = await self.data.basis_rate(symbol)
        tags = obs.strategy
        if (
            basis < -self.BASIS_RATE_THRESHOLD
            if is_long
            else basis > self.BASIS_RATE_THRESHOLD
        ):
            tags = (*tags, StrategyTag.BASIS)
        obs = obs.model_copy(
            update={
                "strategy": tags,
                "side": OrderSide.BUY if is_long else OrderSide.SELL,
            }
        )
        held = Position(
            take_profit=target,
            stop_loss=stop,
            position_side=side,
            entry_price=bars.price,
            name=symbol,
            date=int(context.now.strftime("%Y%m%d")),
            strategy=tags,
            guard=OIStop(open_interest=threshold)
            if is_long and threshold is not None
            else OIStop(open_interest=0),
            long_short_ratio=f"{ratio:.4f}" if ratio is not None else "",
            basis_rate=f"{basis:.4%}",
        )
        return Decision(
            intent=TradeIntent(
                ActionKind.OPEN,
                symbol,
                held,
                bars.price,
                atr,
                observation=obs,
                signal_return=atr * self.TAKE_PROFIT_ATR_FACTOR / bars.price,
            )
        )

    async def manage_position(self, symbol, bars, state, context):
        held = state.positions[symbol]
        is_long = held.position_side == PositionSide.LONG
        price, atr = bars.price, self.calculate_atr(bars)
        upper, lower = (
            self.calc_stop_profit_loss(bars.midprice, is_long, atr)
            if atr > 0
            else (bars.midprice, bars.midprice)
        )
        patch = StatePatch()
        if (held.take_profit == 0 or held.stop_loss == 0) and atr > 0:
            held = held.model_copy(
                update={
                    "take_profit": upper,
                    "stop_loss": lower
                    if not is_long or held.stop_loss == 0
                    else held.stop_loss,
                }
            )
            patch = StatePatch(positions=((symbol, held),))
        reason = None
        ratio = 1.0
        stage = None
        if (
            (held.tp_count > 0 and held.stop_loss > 0 and price <= held.stop_loss)
            if is_long
            else price >= held.stop_loss
        ):
            reason = "止盈后价格止损" if is_long else held.close_reason or "初始止损"
        else:
            tp = price >= held.take_profit if is_long else price <= held.take_profit
            first = False
            if held.tp_count < 1:
                direction = 1 if is_long else -1
                target = self._first_take_profit_target(
                    held.entry_price,
                    (held.take_profit - held.entry_price) * direction,
                    atr,
                )
                first = (
                    target > 0 and (price - held.entry_price) * direction >= target
                ) or (is_long and tp)
            if first or tp:
                reason = "首次止盈" if first else "止盈"
                ratio = self.PARTIAL_CLOSE_RATIO
                stage = "first" if first else "regular"
            elif (
                is_long
                and isinstance(held.guard, OIStop)
                and math.isfinite(held.guard.open_interest)
                and held.guard.open_interest > 0
            ):
                oi = await self.data.oi_5m(symbol)
                if oi and float(oi[-1]["sumOpenInterest"]) <= held.guard.open_interest:
                    reason = "OI止损"
        if reason:
            return Decision(
                patch,
                TradeIntent(
                    ActionKind.CLOSE,
                    symbol,
                    held.model_copy(update={"close_reason": reason}),
                    price,
                    atr,
                    ratio,
                    stage,
                ),
            )
        if symbol in self.data.recovery_symbols():
            self.data.logger.error(
                f"{symbol} 存在未解决的账户方向冲突，仅保留退出检查，暂停加仓"
            )
            return Decision(patch)
        if is_long and not (
            isinstance(held.guard, OIStop)
            and math.isfinite(held.guard.open_interest)
            and held.guard.open_interest > 0
        ):
            try:
                latest, threshold = await self._recovery_oi_threshold(
                    symbol, context.now
                )
                held = held.model_copy(
                    update={"guard": OIStop(open_interest=threshold)}
                )
                patch = StatePatch(positions=((symbol, held),))
                self.data.logger.warning(
                    f"{symbol} 缺失OI止损阈值已按当前数据恢复:{threshold}，不是历史开仓阈值"
                )
                if latest <= threshold:
                    return Decision(
                        patch,
                        TradeIntent(
                            ActionKind.CLOSE,
                            symbol,
                            held.model_copy(update={"close_reason": "OI止损"}),
                            price,
                            atr,
                        ),
                    )
            except Exception as error:
                self.data.logger.error(
                    f"{symbol} OI止损阈值恢复失败，暂停加仓并待下轮重试:{error}"
                )
                return Decision(patch)
        if (
            price < held.entry_price - atr
            if is_long
            else price > held.entry_price + atr
        ):
            return Decision(
                patch, TradeIntent(ActionKind.ADD, symbol, held, price, atr)
            )
        direction = 1 if is_long else -1
        gap = (held.take_profit - held.entry_price) * direction
        decay = gap * self.STOP_LOSS_DECAY_PER_MINUTE * (1 + held.tp_count)
        candidates = [held.take_profit - direction * decay, upper]
        if held.dca_count:
            candidates.append(
                held.entry_price
                + direction * atr * self.DCA_TP_ATR_RATIO**held.dca_count
            )
        target = (
            max(min(candidates), held.entry_price)
            if is_long
            else min(max(candidates), held.entry_price)
        )
        updates = {"take_profit": target}
        if not is_long:
            profit = held.entry_price - price
            protect = held.tp_count > 0 or profit >= self._first_take_profit_target(
                held.entry_price, gap, atr
            )
            stop = (
                held.entry_price - profit * self.TRAILING_STOP_PROFIT_RATIO
                if protect
                else lower
            )
            stop = min(held.stop_loss, stop)
            if stop != held.stop_loss:
                updates.update(
                    stop_loss=stop,
                    close_reason=(
                        "止盈后追踪止损" if held.tp_count > 0 else "追踪止损(保护盈利)"
                    )
                    if protect
                    else "移动止损(轨道)",
                )
        return Decision(
            StatePatch(positions=((symbol, held.model_copy(update=updates)),))
        )

    @staticmethod
    def _inherit_same_side_cooldown(new, previous):
        if previous is None or previous.side != new.side:
            return new
        return new.model_copy(
            update={"earliest_open_timestamp": previous.earliest_open_timestamp}
        )

    async def after_instrument(self, symbol, bars, state, prior, context):
        obs = state.observations.get(symbol)
        refresh_existing_bd = (
            symbol not in prior.positions
            and symbol not in state.positions
            and obs is not None
            and StrategyTag.BD in obs.strategy
        )
        delete_existing_bd = refresh_existing_bd and bars.volumes[-1] >= max(
            bars.volumes[:-1]
        )

        if bars.closes[-2] < bars.price and bars.volumes[-1] >= max(
            bars.volumes[-self.VOLUME_LOOKBACK_PERIOD :]
        ):
            new = Observation(
                price=bars.price,
                timestamp=context.now.timestamp(),
                side=OrderSide.BUY,
                strategy=(StrategyTag.BZ,),
                name=symbol,
            )
            new = self._inherit_same_side_cooldown(
                new, None if delete_existing_bd else obs
            )
            return StatePatch(observations=((symbol, new),))

        if delete_existing_bd:
            return StatePatch(observations=((symbol, None),))

        initial_bd, refresh_bd = await self._evaluate_bd_observation(
            symbol,
            bars.closes,
            bars.volumes,
            context.now,
            refresh_needed=bool(refresh_existing_bd),
        )
        if initial_bd:
            new = Observation(
                price=bars.price,
                timestamp=context.now.timestamp(),
                side=OrderSide.SELL,
                strategy=(StrategyTag.BD,),
                name=symbol,
            )
            new = self._inherit_same_side_cooldown(new, obs)
            return StatePatch(observations=((symbol, new),))

        if refresh_existing_bd and refresh_bd:
            refreshed = obs.model_copy(
                update={
                    "price": bars.price,
                    "timestamp": context.now.timestamp(),
                }
            )
            return StatePatch(observations=((symbol, refreshed),))

        return StatePatch()

    def observation_after_open(self, intent):
        return intent.observation.model_copy(
            update={"strategy": intent.observation.strategy[:1]}
        )

    def observation_after_close(self, intent, state, now):
        if StrategyTag.BD in intent.position.strategy:
            return None
        obs = state.observations.get(intent.symbol)
        return (
            obs.model_copy(
                update={
                    "earliest_open_timestamp": now.timestamp()
                    + self.REOPEN_COOLDOWN_SECONDS
                }
            )
            if obs
            else None
        )

    def position_after_add(self, intent, result):
        held = intent.position
        average = (
            result.average_price
            if result.average_price is not None
            else held.entry_price
        )
        direction = 1 if held.position_side == PositionSide.LONG else -1
        target = average + direction * intent.atr * self.DCA_TP_ATR_RATIO ** (
            held.dca_count + 1
        )
        target = (
            min(held.take_profit, target)
            if direction == 1
            else max(held.take_profit, target)
        )
        return held.model_copy(
            update={
                "entry_price": average,
                "strategy": (*held.strategy, StrategyTag.DCA),
                "take_profit": target,
            }
        )

    def position_after_take_profit(self, intent, result):
        held = intent.position
        target, stop = held.take_profit, held.stop_loss
        is_long = held.position_side == PositionSide.LONG
        if intent.atr > 0:
            upper, lower = self.calc_stop_profit_loss(intent.price, is_long, intent.atr)
            if upper > 0 and lower > 0:
                target = upper
                if not is_long:
                    stop = lower
        if intent.take_profit_stage == "first":
            distance = min(intent.atr, intent.price * self.ATR_TRIGGER_CAP_RATIO)
            if is_long:
                target = intent.price + distance
            else:
                stop = min(
                    stop,
                    held.entry_price
                    - (intent.price - held.entry_price)
                    * -1
                    * self.TRAILING_STOP_PROFIT_RATIO,
                )
                target = intent.price - distance
        return held.model_copy(
            update={
                "take_profit": target,
                "stop_loss": stop,
                "tp_count": held.tp_count + (intent.take_profit_stage is not None),
            }
        )

    def signal_message(self, intent):
        held = intent.position
        return (
            f"==={intent.symbol}**{format_strategy_tags(held.strategy)}**===\n价格:{intent.price}\n"
            f"基差率:{held.basis_rate}\n多空比:{held.long_short_ratio or 'N/A'}\n"
            f"止盈:{held.take_profit}\n止损:{held.stop_loss}\n收益率:{intent.signal_return:.2%}"
        )

    def trade_message(self, order, result, position):
        intent = order.intent
        if intent.kind == ActionKind.CLOSE:
            pnl = (intent.price - result.entry_price) * (
                1 if position.position_side == PositionSide.LONG else -1
            )
            msg = (
                f"{intent.symbol} 平仓\n策略:{format_strategy_tags(position.strategy)}\n持仓方向:{position.position_side.value}"
                f"\n委托价格:{intent.price}\n委托数量:{result.original_quantity}\n平仓比例:{result.ratio:.2%}"
                f"\n平仓盈亏:{pnl * result.quantity} USDT\n平仓收益:{pnl / result.entry_price if result.entry_price else 0:.2%}"
                f"\n止盈次数:{position.tp_count}\n平仓依据:{intent.position.close_reason}"
            )
            return format_trade_notification("AUTOBN", "平仓", intent.symbol, msg)
        tags = (
            (*intent.position.strategy, StrategyTag.DCA)
            if intent.kind == ActionKind.ADD
            else intent.position.strategy
        )
        rate = abs(intent.position.take_profit - order.price) / order.price
        msg = (
            f"{intent.symbol} 开仓\n策略:{format_strategy_tags(tags)}\n持仓方向:{intent.position.position_side.value}"
            f"\n杠杆:{order.leverage}x\n委托数量:{result.original_quantity}\n委托价格:{order.price}\n名义价值:{order.notional} USDT"
            f"\n账户余额:{order.account_equity:.2f}\n仓位比例:{order.notional / order.account_equity:.2%}\n收益率:{rate:.2%}"
        )
        return format_trade_notification("AUTOBN", "开仓", intent.symbol, msg)

    async def check_side(
        self,
        symbol,
        positionSide,
        dtn: datetime = None,
    ):
        """
        检查增仓信号，判断是否适合开仓
        """
        try:
            if positionSide == PositionSide.SHORT.value:
                realtime_oi, completed_oi = await self.data.bd_oi_windows(symbol, dtn)
                if realtime_oi is None:
                    return False, None, None
                oi_5m_last = realtime_oi[-1]
                oi_peak = max(completed_oi)
                passed = oi_5m_last <= oi_peak * (1 - self.BD_OI_DRAWDOWN_RATIO)
                return (True, None, None) if passed else (False, None, None)

            # 多空人数比：1h序列按UTC小时缓存，5m那根按当前周期缓存
            lsr_1h = await self.data.lsr_1h(symbol, dtn)
            lsr_5m = await self.data.lsr_5m(symbol)
            if not lsr_1h or not lsr_5m:
                return False, None, None

            lsrd = float(lsr_5m[0]["longShortRatio"])  # 当前值

            # 多仓比例提取：优先 longAccount，缺失时由 longShortRatio 推导
            def _extract_long_ratio(item):
                long_account = item.get("longAccount")
                if long_account is not None:
                    return float(long_account)
                lsr = float(item["longShortRatio"])
                return lsr / (1 + lsr)

            oi_5m = await self.data.oi_5m(symbol)

            if not oi_5m:
                return False, None, None
            oi_5m_last = float(oi_5m[-1]["sumOpenInterest"])
            if positionSide == PositionSide.LONG.value:
                oi_1h = await self.data.oi_history(symbol, dtn, "1h")

                if not oi_1h:
                    return False, None, None
                sum_open_interest_1h = [
                    float(item["sumOpenInterest"]) for item in oi_1h
                ]
                oi_hist_for_cheb = sum_open_interest_1h[
                    : -self.LONG_OI_CHEB_EXCLUDE_RECENT_COUNT
                ]
                if len(oi_hist_for_cheb) < self.MIN_CHEB_SAMPLE_SIZE:
                    return False, None, None

                if oi_5m_last <= 0:
                    return False, None, None

                # blend 基准：切比雪夫区间最后一根的OI和多仓比例（同一根）
                # 两条1h序列独立更新缓存，末根可能错位，按时间戳配对
                window_end_oi = oi_hist_for_cheb[-1]
                window_end_ts = int(
                    oi_1h[-self.LONG_OI_CHEB_EXCLUDE_RECENT_COUNT - 1]["timestamp"]
                )
                lsr_1h_by_ts = {int(item["timestamp"]): item for item in lsr_1h}
                window_end_lsr = lsr_1h_by_ts.get(window_end_ts)
                if window_end_lsr is None:
                    return False, None, None

                window_end_long_ratio = _extract_long_ratio(window_end_lsr)
                long_ratio_5m = _extract_long_ratio(lsr_5m[0])

                blend = (
                    window_end_oi * window_end_long_ratio
                    + (oi_5m_last - window_end_oi) * self.OI_DELTA_LONG_RATIO_WEIGHT
                ) / oi_5m_last

                if blend < long_ratio_5m:
                    return False, None, None

                chebyshev = sample_probability(
                    oi_hist_for_cheb,
                    oi_5m_last,
                )
                passed = (
                    oi_5m_last >= max(sum_open_interest_1h)
                    and chebyshev["chebyshev_upper_bound"]
                    < self.CHEBYSHEV_EXTREME_THRESHOLD
                )
                if not passed:
                    return False, None, None

                stop_guard_threshold = self._oi_stop_threshold(oi_5m_last, chebyshev)
                return True, lsrd, stop_guard_threshold

        except Exception:
            self.data.logger.exception("检查增仓信号时发生错误")
            return False, None, None

    def _oi_stop_threshold(self, peak_oi, statistics):
        threshold = max(
            peak_oi * (1 - self.BZ_LONG_OI_DRAWDOWN_RATIO),
            statistics["mean"]
            + statistics["std"] / self.CHEBYSHEV_EXTREME_THRESHOLD**0.5,
        )
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("计算出的OI止损阈值无效")
        return threshold

    async def _recovery_oi_threshold(self, symbol, now):
        current = await self.data.oi_5m(symbol)
        history = await self.data.oi_history(symbol, now, "1h")
        if not current or not history or len(history) < self.OI_QUERY_LIMIT:
            raise ValueError("恢复OI阈值所需数据不足")
        latest = float(current[-1]["sumOpenInterest"])
        values = [float(row["sumOpenInterest"]) for row in history]
        if any(not math.isfinite(value) or value <= 0 for value in [latest, *values]):
            raise ValueError("恢复OI阈值的数据无效")
        sample = values[: -self.LONG_OI_CHEB_EXCLUDE_RECENT_COUNT]
        return latest, self._oi_stop_threshold(
            max(latest, max(values)), sample_probability(sample, latest)
        )

    async def _recover_unregistered_position(self, symbol, bars, state, context):
        try:
            rows = await self.data.gateway.positions(symbol)
            if not rows:
                self.data.resolve_position_recovery(symbol)
                return Decision()
            if (
                len(rows) != 1
                or rows[0].get("symbol") != symbol
                or rows[0]["positionSide"] not in ("LONG", "SHORT")
            ):
                raise ValueError("当前单币单仓模型无法无歧义恢复该账户仓位")
            row = rows[0]
            side = PositionSide(row["positionSide"])
            entry = float(row["entryPrice"])
            atr = self.calculate_atr(bars)
            target, stop = self.calc_stop_profit_loss(
                bars.midprice, side == PositionSide.LONG, atr
            )
            if not all(math.isfinite(x) and x > 0 for x in (entry, atr, target, stop)):
                raise ValueError("恢复仓位的均价/ATR轨道无效")
            latest, threshold = (
                await self._recovery_oi_threshold(symbol, context.now)
                if side == PositionSide.LONG
                else (None, None)
            )
            held = Position(
                take_profit=target,
                stop_loss=stop,
                position_side=side,
                entry_price=entry,
                name=symbol,
                date=int(context.now.strftime("%Y%m%d")),
                strategy=(StrategyTag.N,),
                guard=OIStop(open_interest=threshold)
                if threshold is not None
                else None,
            )
            patch = StatePatch(positions=((symbol, held),))
            self.data.resolve_position_recovery(symbol)
            self.data.logger.warning(
                f"{symbol} 实际仓位已重建为N；使用当前风控基准，未知历史DCA/止盈次数不回填"
            )
            if latest is not None and latest <= threshold:
                return Decision(
                    patch,
                    TradeIntent(
                        ActionKind.CLOSE,
                        symbol,
                        held.model_copy(update={"close_reason": "OI止损"}),
                        bars.price,
                        atr,
                    ),
                )
            return Decision(patch)
        except Exception as error:
            self.data.logger.error(
                f"{symbol} N持仓恢复失败，禁止新开仓并待下轮重试:{error}"
            )
            return Decision()

    async def _evaluate_bd_observation(
        self,
        symbol,
        kline_close,
        kline_volume,
        dtn,
        *,
        refresh_needed=False,
    ):
        """返回（首次入池，刷新已有BD）；行情和OI各只评估一次。"""
        if kline_close[-1] < max(kline_close[:-1]):
            self.data.gateway.diagnostics.note("prefilter", "bd", "", "rejected")
            return False, False

        volume_peak = max(kline_volume)
        if max(kline_volume[-self.BD_VOLUME_RECENT_COUNT :]) >= volume_peak:
            self.data.gateway.diagnostics.note("prefilter", "bd", "", "rejected")
            return False, False

        initial_market_match = (
            max(
                kline_volume[
                    -self.BD_VOLUME_LOOKBACK_COUNT : -self.BD_VOLUME_RECENT_COUNT
                ]
            )
            == volume_peak
        )
        if not (initial_market_match or refresh_needed):
            self.data.gateway.diagnostics.note("prefilter", "bd", "", "rejected")
            return False, False
        self.data.gateway.diagnostics.note("prefilter", "bd", "", "passed")

        oi_window, _ = await self.data.bd_oi_windows(symbol, dtn)
        if oi_window is None or len(oi_window) < self.OI_QUERY_LIMIT:
            return False, False

        oi_baseline = oi_window[: -self.BD_OI_LOOKBACK_COUNT]
        oi_old = oi_window[-self.BD_OI_LOOKBACK_COUNT : -self.BD_OI_RECENT_COUNT]
        oi_old_peak = max(oi_old)
        if oi_window[-1] <= oi_old_peak:
            return False, False

        initial = initial_market_match and (
            sample_probability(oi_baseline, oi_old_peak)["chebyshev_upper_bound"]
            < self.SHORT_OI_CHEB_THRESHOLD
        )
        return initial, True

    def calc_stop_profit_loss(self, price, is_long=True, atr=0):
        if atr <= 0:
            return (0, 0)

        take_profit_distance = atr * self.TAKE_PROFIT_ATR_FACTOR
        stop_loss_distance = atr * self.STOP_LOSS_ATR_FACTOR
        if is_long:
            return (
                price + take_profit_distance,
                price - stop_loss_distance,
            )
        else:
            # 做空：返回 (下界/止盈位, 上界/止损位)
            return (
                price - take_profit_distance,
                price + stop_loss_distance,
            )

    def _first_take_profit_target(self, entry_price, take_profit_gap, atr_value):
        atr_trigger = max(
            min(atr_value, entry_price * self.ATR_TRIGGER_CAP_RATIO),
            self.MIN_ATR_TRIGGER,
        )
        return min(take_profit_gap / self.TARGET_PROFIT_DIVISOR, atr_trigger)

    def calculate_atr(self, bars):
        period = self.ATR_PERIOD
        if len(bars) < period + 1:
            return 0.0
        tr = [
            max(
                bars.highs[i] - bars.lows[i],
                abs(bars.highs[i] - bars.closes[i - 1]),
                abs(bars.lows[i] - bars.closes[i - 1]),
            )
            for i in range(1, len(bars))
        ]
        atr = sum(tr[:period]) / period
        for value in tr[period:]:
            atr = (atr * (period - 1) + value) / period
        cap = bars.midprice * self.ATR_HL2_CAP_RATIO
        return min(atr, cap) if cap > 0 else atr


class AUTOA(MarketStrategy):
    MARKET = "AUTOA"
    SIGNAL_CHANNEL = "wecom"
    CYCLE_HOURS = "09-15"
    WEEKDAYS = "mon-fri"
    DAILY_HOUR = 15
    DAILY_MINUTE = 6
    OBSERVATION_TIMEOUT_SECONDS = 30 * 86400
    ATR_PERIOD = 5
    TAKE_PROFIT_ATR_FACTOR = 3.0
    STOP_LOSS_ATR_FACTOR = 1.0
    ATR_TRIGGER_CAP_RATIO = 0.05
    ATR_HL2_CAP_RATIO = 0.1
    MIN_ATR_TRIGGER = 1e-8
    DCA_TP_ATR_RATIO = 0.5
    TAKE_PROFIT_DECAY = 0.001
    TARGET_PROFIT_DIVISOR = 3.0
    VOLUME_CHEB_REQUIRED_HISTORY = 20
    VOLUME_CHEB_SAMPLE_START_OFFSET = -20
    VOLUME_CHEB_SAMPLE_END_OFFSET = -5
    CHEBYSHEV_EXTREME_THRESHOLD = 0.01

    async def prepare_cycle(self, now, state):
        return await self.data.context(now), StatePatch()

    def scan_candidates(self, state):
        return tuple(code for code in state.observations if code not in state.positions)

    async def daily_report_due(self, now):
        return await self.data.is_trading_day(now)

    async def prepare_instrument(self, symbol, state, context):
        if symbol in state.positions:
            return Decision()
        obs = state.observations.get(symbol)
        if obs is None or context.exit_only:
            return Decision(proceed=False)
        patch = StatePatch()
        if obs.reopen_pending_date is not None:
            if context.calendar_request is None:
                context.calendar_request = asyncio.create_task(
                    self.data.trading_calendar()
                )
            calendar = await context.calendar_request
            if calendar is None:
                return Decision(proceed=False)
            reopen = await self.data.reopen_timestamp(
                pd.Timestamp(obs.reopen_pending_date).to_pydatetime(), calendar=calendar
            )
            if reopen is None:
                return Decision(proceed=False)
            obs = obs.model_copy(
                update={"earliest_open_timestamp": reopen, "reopen_pending_date": None}
            )
            patch = StatePatch(observations=((symbol, obs),))
        now = context.now.timestamp()
        if obs.is_reopen_cooldown_active(now):
            return Decision(patch, proceed=False)
        if now - obs.timestamp > self.OBSERVATION_TIMEOUT_SECONDS:
            return Decision(StatePatch(observations=((symbol, None),)), proceed=False)
        if (
            datetime.datetime.fromtimestamp(now, datetime.timezone.utc).date()
            == datetime.datetime.fromtimestamp(
                obs.timestamp, datetime.timezone.utc
            ).date()
        ):
            return Decision(patch, proceed=False)
        return Decision(patch)

    def update_observation(self, obs, bars):
        if not bars or bars.lows[-1] <= 0 or math.isnan(bars.lows[-1]):
            return obs, False
        if StrategyTag.BZ in obs.strategy:
            obs = obs.model_copy(update={"price": min(obs.price, bars.lows[-1])})
        if len(bars) < self.VOLUME_CHEB_REQUIRED_HISTORY or not (
            bars.price > bars.opens[-1] and bars.opens[-1] > bars.highs[-2]
        ):
            return obs, False
        sample = bars.volumes[
            self.VOLUME_CHEB_SAMPLE_START_OFFSET : self.VOLUME_CHEB_SAMPLE_END_OFFSET
        ]
        triggered = (
            bars.volumes[-1] >= max(bars.volumes[self.VOLUME_CHEB_SAMPLE_END_OFFSET :])
            and sample_probability(sample, bars.volumes[-1], pandas_sample=True)[
                "chebyshev_upper_bound"
            ]
            < self.CHEBYSHEV_EXTREME_THRESHOLD
        )
        if triggered:
            obs = obs.model_copy(
                update={
                    "price": bars.lows[-1],
                    "bz_reference_high": bars.highs[-2],
                    "strategy": obs.strategy
                    if StrategyTag.BZ in obs.strategy
                    else (*obs.strategy, StrategyTag.BZ),
                }
            )
        return obs, triggered

    def calculate_atr(self, bars):
        period = self.ATR_PERIOD
        if len(bars) < period + 1:
            return 0.0
        tr = [
            max(
                bars.highs[i] - bars.lows[i],
                abs(bars.highs[i] - bars.closes[i - 1]),
                abs(bars.lows[i] - bars.closes[i - 1]),
            )
            for i in range(len(bars) - period, len(bars))
        ]
        atr = sum(tr) / period
        cap = bars.midprice * self.ATR_HL2_CAP_RATIO
        return min(atr, cap) if cap > 0 else atr

    async def evaluate_signal(self, symbol, bars, state, context):
        obs = state.observations.get(symbol)
        if obs is None:
            return Decision()
        obs, bz = self.update_observation(obs, bars)
        patch = StatePatch(observations=((symbol, obs),))
        if (
            len(bars) < self.VOLUME_CHEB_REQUIRED_HISTORY
            or bars.price <= bars.opens[-1]
        ):
            return Decision(patch)
        n = (
            not bz
            and StrategyTag.BZ in obs.strategy
            and obs.bz_reference_high is not None
            and obs.bz_reference_high > 0
            and bars.lows[-2] < obs.bz_reference_high
            and bars.opens[-1] > bars.highs[-2]
        )
        if not (bz or n):
            return Decision(patch)
        atr = self.calculate_atr(bars)
        if not (math.isfinite(atr) and atr > 0):
            return Decision(patch)
        if n and StrategyTag.N not in obs.strategy:
            obs = obs.model_copy(update={"strategy": (*obs.strategy, StrategyTag.N)})
        sample = bars.volumes[
            self.VOLUME_CHEB_SAMPLE_START_OFFSET : self.VOLUME_CHEB_SAMPLE_END_OFFSET
        ]
        target = bars.midprice + atr * self.TAKE_PROFIT_ATR_FACTOR
        stop = (
            obs.price
            if StrategyTag.N in obs.strategy
            else bars.midprice - atr * self.STOP_LOSS_ATR_FACTOR
        )
        held = Position(
            take_profit=target,
            stop_loss=stop,
            position_side=PositionSide.LONG,
            entry_price=bars.price,
            name=obs.name,
            date=int(context.now.strftime("%Y%m%d")),
            strategy=obs.strategy,
            guard=VolumeStop(volume=float(sum(sample) / len(sample))),
        )
        return Decision(
            patch,
            TradeIntent(
                ActionKind.OPEN,
                symbol,
                held,
                bars.price,
                atr,
                observation=obs,
                signal_return=abs(target - bars.price) / bars.price
                if bars.price > 0
                else 0,
            ),
        )

    async def manage_position(self, symbol, bars, state, context):
        held = state.positions[symbol]
        obs = state.observations.get(symbol)
        patch = StatePatch()
        if obs:
            obs, _ = self.update_observation(obs, bars)
            patch = StatePatch(observations=((symbol, obs),))
        if (
            int(context.now.strftime("%Y%m%d")) <= held.date
            or len(bars) < self.VOLUME_CHEB_REQUIRED_HISTORY
        ):
            return Decision(patch)
        n = StrategyTag.N in held.strategy
        reason = None
        if bars.price >= held.take_profit:
            reason = "止盈"
        elif bars.price <= held.stop_loss:
            reason = held.close_reason or "初始止损"
        elif (
            not n
            and isinstance(held.guard, VolumeStop)
            and 0 < bars.volumes[-2] <= held.guard.volume
        ):
            reason = "成交量止损"
        if reason:
            return Decision(
                patch,
                TradeIntent(
                    ActionKind.CLOSE,
                    symbol,
                    held.model_copy(update={"close_reason": reason}),
                    bars.price,
                ),
            )
        atr = self.calculate_atr(bars)
        gap = held.take_profit - held.entry_price
        trigger = max(
            min(atr, held.entry_price * self.ATR_TRIGGER_CAP_RATIO),
            self.MIN_ATR_TRIGGER,
        )
        if gap > 0 and bars.price - held.entry_price >= min(
            gap / self.TARGET_PROFIT_DIVISOR, trigger
        ):
            return Decision(
                patch,
                TradeIntent(
                    ActionKind.CLOSE,
                    symbol,
                    held.model_copy(update={"close_reason": "首次止盈"}),
                    bars.price,
                ),
            )
        if context.exit_only:
            return Decision(patch)
        if atr > 0 and held.entry_price > 0 and bars.price < held.entry_price - atr:
            return Decision(
                patch, TradeIntent(ActionKind.ADD, symbol, held, bars.price, atr)
            )
        target = min(
            held.take_profit - gap * self.TAKE_PROFIT_DECAY,
            bars.midprice + atr * self.TAKE_PROFIT_ATR_FACTOR,
        )
        if held.dca_count:
            target = min(
                target, held.entry_price + atr * self.DCA_TP_ATR_RATIO**held.dca_count
            )
        updates = {"take_profit": max(target, held.entry_price)}
        if not n:
            stop = max(held.stop_loss, bars.midprice - atr * self.STOP_LOSS_ATR_FACTOR)
            if stop != held.stop_loss:
                updates.update(stop_loss=stop, close_reason="移动止损(轨道)")
        return Decision(
            replace(patch, positions=((symbol, held.model_copy(update=updates)),))
        )

    async def refresh_universe(self, now, state):
        pool, context = await self.data.limit_up_pool(now)
        if not pool:
            return False, StatePatch()
        semaphore = asyncio.Semaphore(8)

        async def build(code, name):
            async with semaphore:
                try:
                    bars = await self.data.fetch_bars(code, context)
                    if not bars:
                        return None
                    obs = state.observations.get(code)
                    if obs:
                        if StrategyTag.BZ in obs.strategy:
                            obs, _ = self.update_observation(obs, bars)
                        obs = obs.model_copy(update={"timestamp": now.timestamp()})
                    else:
                        obs = Observation(
                            price=bars.price,
                            timestamp=now.timestamp(),
                            side=OrderSide.BUY,
                            strategy=(),
                            name=name,
                        )
                    return code, obs
                except Exception:
                    self.data.logger.exception(f"{code}收盘刷新失败")
                    return None

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(build(code, name)) for code, name in pool]
        changes = tuple(t.result() for t in tasks if t.result() is not None)
        self.data.invalidate_history()
        return True, StatePatch(observations=changes)

    async def build_daily_report(self, now, state):
        rows = []
        total = 0
        for code, held in state.positions.items():
            bar = await self.data.daily_bar(code, now=now)
            common = dict(
                name=f"{held.name}({code})",
                strategy=format_strategy_tags(held.strategy),
                direction=held.position_side.value,
                entry_price=held.entry_price,
                take_profit=held.take_profit,
                stop_loss=held.stop_loss,
                open_date=held.date,
                currency="CNY",
                price_decimals=2,
            )
            if bar is None:
                rows.append(
                    DailyPosition(
                        **common, note="行情获取失败，暂不可估值（未计入总持仓金额）。"
                    )
                )
                continue
            shares = self.SHARES_PER_ORDER * (1 + held.dca_count)
            notional = bar["close"] * shares
            cost = held.entry_price * shares
            total += notional
            rows.append(
                DailyPosition(
                    **common,
                    quantity=shares,
                    current_price=bar["close"],
                    notional=notional,
                    unrealized_pnl=notional - cost,
                    profit_rate=(notional - cost) / cost if cost > 0 else 0,
                )
            )
        return format_daily_positions_notification(
            self.MARKET, "总持仓金额", total, "CNY", rows
        )

    def observation_after_close(self, intent, state, now):
        obs = state.observations.get(intent.symbol) or Observation(
            price=intent.price,
            timestamp=now.timestamp(),
            side=OrderSide.BUY,
            strategy=(),
            name=intent.position.name,
        )
        return obs.model_copy(
            update={
                "earliest_open_timestamp": None,
                "reopen_pending_date": now.strftime("%Y-%m-%d"),
            }
        )

    def position_after_add(self, intent, result):
        held = intent.position
        average = (
            result.average_price
            if result.average_price is not None
            else held.entry_price
        )
        target = min(
            held.take_profit,
            average + intent.atr * self.DCA_TP_ATR_RATIO ** (held.dca_count + 1),
        )
        return held.model_copy(
            update={
                "entry_price": average,
                "take_profit": target,
                "strategy": (*held.strategy, StrategyTag.DCA),
            }
        )

    def position_after_take_profit(self, intent, result):
        return intent.position

    def record_symbol(self, symbol, position):
        return f"{position.name} {symbol}"

    def signal_message(self, intent):
        held = intent.position
        return (
            f"==={held.name}**{format_strategy_tags(held.strategy)}**===\n价格:{intent.price:.2f}\n"
            f"止盈:{held.take_profit:.2f}\n止损:{held.stop_loss:.2f}\n收益率:{intent.signal_return:.2%}\n"
        )

    def trade_message(self, order, result, position):
        intent = order.intent
        if StrategyTag.N not in position.strategy:
            return None
        if intent.kind == ActionKind.OPEN:
            return format_trade_notification(
                self.MARKET, "开仓", position.name, self.signal_message(intent)
            )
        if intent.kind == ActionKind.ADD:
            msg = (
                f"{position.name} 加仓\n策略:{format_strategy_tags(position.strategy)}\n委托价格:{intent.price:.2f}\n"
                f"止盈:{position.take_profit:.2f}\n止损:{position.stop_loss:.2f}"
            )
            return format_trade_notification(self.MARKET, "开仓", position.name, msg)
        profit = intent.price / result.entry_price - 1 if result.entry_price > 0 else 0
        pnl = (intent.price - result.entry_price) * result.quantity
        msg = (
            f"{position.name} 平仓\n策略:{format_strategy_tags(position.strategy)}\n持仓方向:{position.position_side.value}\n"
            f"委托价格:{intent.price:.2f}\n委托数量:{result.quantity}\n平仓盈亏:{pnl:.2f} CNY\n"
            f"平仓收益:{profit:.2%}\n平仓依据:{intent.position.close_reason}"
        )
        return format_trade_notification(self.MARKET, "平仓", position.name, msg)


def create_engine(market, *, state_file=None, records=None, gateway=None):
    logger = logging.getLogger(f"{market}.{time.monotonic_ns()}")
    logger.setLevel(logging.INFO)
    http = HttpSession()
    if market == "AUTOBN":
        gateway = gateway or BinanceGateway()
        data = BinanceMarketData(gateway, logger)
        strategy = AUTOBN(data)
        executor = BinanceExecutor(gateway, data, logger)
        filename = "alert_all.json"
    elif market == "AUTOA":
        data = AshareMarketData(http, logger)
        strategy = AUTOA(data)
        executor = AshareExecutor()
        filename = "alert_all_A.json"
    else:
        raise ValueError(f"未知市场:{market}")
    state = TradingState(
        market, state_file or os.path.join(os.path.dirname(__file__), filename)
    )
    notifications = Notifications(http, logger)
    if market == "AUTOBN":
        data.set_disconnect_notifier(
            lambda message: notifications.send(
                TradeNotification("币安K线WS连接异常", message), channel="pushplus"
            )
        )
    return TradingEngine(
        strategy,
        executor,
        state,
        notifications,
        records if records is not None else CloseRecordManager(),
        http,
        logger,
    )


def handle_exit_signal(signum, _frame):
    logging.getLogger("AutoTrade").info(f"退出信号:{signum}")
    sys.exit(0)


async def main():
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    records = CloseRecordManager()
    engines = (
        create_engine("AUTOA", records=records),
        create_engine("AUTOBN", records=records),
    )
    for engine in engines:
        engine.initialize()
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)
    scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    for engine in engines:
        strategy = engine.strategy
        scheduler.add_job(
            engine.run_market_cycle,
            "cron",
            hour=strategy.CYCLE_HOURS,
            minute="*",
            second=0,
            day_of_week=strategy.WEEKDAYS,
            misfire_grace_time=10,
            max_instances=1,
            coalesce=True,
            name=f"{strategy.MARKET} 行情扫描",
        )
        scheduler.add_job(
            engine.push_daily_report,
            "cron",
            hour=strategy.DAILY_HOUR,
            minute=strategy.DAILY_MINUTE,
            second=0,
            day_of_week=strategy.WEEKDAYS,
            misfire_grace_time=300,
            max_instances=1,
            coalesce=True,
            name=f"{strategy.MARKET} 每日持仓",
        )
    scheduler.start()
    try:
        await asyncio.Event().wait()
    finally:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            logging.getLogger("AutoTrade").exception("停止调度失败")
        results = await asyncio.gather(
            *(e.close() for e in engines), return_exceptions=True
        )
        await records.flush_pending_records()
        for result in results:
            if isinstance(result, BaseException):
                logging.getLogger("AutoTrade").error(f"退出清理失败:{result}")


if __name__ == "__main__":
    asyncio.run(main())
