import asyncio
import datetime
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

from perk_pushplus import PushPlusClient, SendRequest, Template


@dataclass(frozen=True)
class TradeNotification:
    title: str
    content: str


@dataclass(frozen=True)
class DailyPosition:
    """每日持仓通知中的单个持仓。"""

    name: str
    strategy: str
    direction: str
    entry_price: float
    take_profit: float
    stop_loss: float
    open_date: int | str
    currency: str
    notional: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    profit_rate: Optional[float] = None
    quantity: Optional[float] = None
    current_price: Optional[float] = None
    price_decimals: int = 8
    note: Optional[str] = None


def format_trade_notification(
    market: str,
    action: str,
    symbol: str,
    message: str,
    success: bool = True,
) -> TradeNotification:
    status_icon = "✅" if success else "❌"
    status_text = "成功" if success else "失败"
    title = f"{symbol} {action}{status_text}"

    lines = [line.strip() for line in message.splitlines() if line.strip()]
    detail_lines = []
    quote_lines = []
    body_lines = lines[1:]
    if not success and len(lines) == 1:
        reason_parts = re.split(r"失败[：:]|跳过[：:]", lines[0], maxsplit=1)
        if len(reason_parts) == 2 and reason_parts[1].strip():
            body_lines = [f"原因:{reason_parts[1].strip()}"]
    for line in body_lines:
        if ":" in line or "：" in line:
            separator = ":" if ":" in line else "："
            label, value = line.split(separator, 1)
            detail_lines.append(f"- **{label.strip()}：** {value.strip()}")
        else:
            quote_lines.append(f"> {line}")

    content_parts = [
        f"# {status_icon} {market} · {action}{status_text}",
        "",
        f"## {symbol}",
        "",
    ]
    if detail_lines:
        content_parts.extend(detail_lines)
    if quote_lines:
        if detail_lines:
            content_parts.extend(["", "---", ""])
        content_parts.extend(quote_lines)
    if not success:
        content_parts.extend(
            [
                "",
                f"- **时间：** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "> 本次交易动作未完成，请检查账户、持仓及运行日志。",
            ]
        )

    return TradeNotification(title=title, content="\n".join(content_parts))


def _format_money(value: float, currency: str) -> str:
    return f"{value:,.2f} {currency}"


def _format_price(value: float, decimals: int) -> str:
    formatted = f"{value:,.{decimals}f}"
    if decimals > 2:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _format_open_date(value: int | str) -> str:
    date_text = str(value)
    if len(date_text) == 8 and date_text.isdigit():
        return f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:]}"
    return date_text


def _change_icon(value: float) -> str:
    if value > 0:
        return "🟢"
    if value < 0:
        return "🔴"
    return "⚪"


def _direction_label(direction: str) -> str:
    normalized = direction.upper()
    if normalized == "LONG":
        return "🟢 LONG"
    if normalized == "SHORT":
        return "🔴 SHORT"
    return direction or "方向未知"


def format_daily_positions_notification(
    market: str,
    summary_label: str,
    summary_value: float,
    currency: str,
    positions: Sequence[DailyPosition],
) -> TradeNotification:
    """将每日持仓排成适合 PushPlus Markdown 阅读的分节卡片。"""
    content_parts = [
        f"# 📊 {market} · 每日持仓",
        "",
        "## 账户概览",
        "",
        f"- **{summary_label}：** `{_format_money(summary_value, currency)}`",
        f"- **持仓数量：** `{len(positions)}`",
    ]

    if not positions:
        content_parts.extend(["", "> 当前暂无持仓。"])

    for index, position in enumerate(positions, start=1):
        content_parts.extend(
            [
                "",
                "---",
                "",
                f"## {index}. {position.name} · {_direction_label(position.direction)}",
                "",
                f"**策略：** `{position.strategy or 'N/A'}`",
                "",
                "- **开仓价格：** "
                f"`{_format_price(position.entry_price, position.price_decimals)} "
                f"{position.currency}`",
            ]
        )
        if position.quantity is not None:
            content_parts.append(f"- **策略持仓股数：** `{position.quantity:,.0f}`")
        if position.current_price is not None:
            content_parts.append(
                "- **竞价行情价：** "
                f"`{_format_price(position.current_price, position.price_decimals)} "
                f"{position.currency}`"
            )
        if position.notional is not None:
            content_parts.append(
                f"- **名义价值：** `{_format_money(position.notional, position.currency)}`"
            )
        if position.unrealized_pnl is not None:
            content_parts.append(
                f"- **持仓盈亏：** {_change_icon(position.unrealized_pnl)} "
                f"**{_format_money(position.unrealized_pnl, position.currency)}**"
            )
        if position.profit_rate is not None:
            content_parts.append(
                f"- **持仓收益：** {_change_icon(position.profit_rate)} "
                f"**{position.profit_rate:.2%}**"
            )
        content_parts.extend(
            [
                "- **止盈 / 止损：** "
                f"`{_format_price(position.take_profit, position.price_decimals)}` / "
                f"`{_format_price(position.stop_loss, position.price_decimals)}`",
                f"- **开仓日期：** `{_format_open_date(position.open_date)}`",
            ]
        )
        if position.note:
            content_parts.extend(["", f"> ⚠️ {position.note}"])

    return TradeNotification(
        title=f"{market} 每日持仓",
        content="\n".join(content_parts),
    )


def classify_autobn_message(message: str) -> Optional[TradeNotification]:
    first_line = message.splitlines()[0].strip() if message.strip() else ""
    if "开仓" in first_line:
        action = "开仓"
    elif "平仓" in first_line:
        action = "平仓"
    else:
        return None

    success = "失败" not in first_line and "跳过" not in first_line
    symbol = first_line.split(action, 1)[0].strip()
    if symbol.lower() == "bn":
        symbol = first_line.split(action, 1)[1]
        symbol = re.split(r"失败|跳过|[：:]", symbol, maxsplit=1)[0].strip()
    symbol = symbol or "未知标的"
    return format_trade_notification("AUTOBN", action, symbol, message, success)


@lru_cache(maxsize=4)
def _get_pushplus_client(token: str, secret_key: str) -> PushPlusClient:
    builder = PushPlusClient.builder().token(token)
    if secret_key:
        builder = builder.secret_key(secret_key)
    return builder.build()


def _send_pushplus_sync(
    notification: TradeNotification,
    token: str,
    secret_key: str,
) -> None:
    request = (
        SendRequest.builder()
        .title(notification.title)
        .content(notification.content)
        .template(Template.MARKDOWN)
        .build()
    )
    _get_pushplus_client(token, secret_key).send(request)


async def send_pushplus(
    notification: TradeNotification,
    token: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> bool:
    target_token = token or os.getenv("PUSHPLUS_TOKEN")
    if not target_token:
        return False

    target_secret_key = secret_key
    if target_secret_key is None:
        target_secret_key = os.getenv("PUSHPLUS_SECRET_KEY", "")
    await asyncio.to_thread(
        _send_pushplus_sync,
        notification,
        target_token,
        target_secret_key,
    )
    return True
