import asyncio
import datetime
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import autoTrade_pm_refactored as trading

SYMBOL = "BTCUSDT"
NOW = datetime.datetime(2026, 9, 17, 12, 0, tzinfo=datetime.UTC)
OI_WINDOW = tuple(
    [1.0] * 21
    + [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0]
    + [30.0, 20.0]
    + [120.0]
)


def make_bars(*, close=100.0, peak_index=22, last_volume=5.0):
    closes = [100.0] * 30
    closes[-1] = close
    volumes = [10.0] * 30
    volumes[peak_index] = 100.0
    volumes[-3:] = [8.0, 7.0, last_volume]
    prices = tuple([100.0] * 30)
    return trading.MarketBars(
        times=tuple(range(30)),
        opens=prices,
        highs=prices,
        lows=prices,
        closes=tuple(closes),
        volumes=tuple(volumes),
    )


def make_observation(**updates):
    values = {
        "price": 90.0,
        "timestamp": 1.0,
        "side": trading.OrderSide.SELL,
        "strategy": (trading.StrategyTag.BD,),
        "name": SYMBOL,
        "earliest_open_timestamp": NOW.timestamp() + 3600,
    }
    values.update(updates)
    return trading.Observation(**values)


class AUTOBNOIThresholdTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.data = Mock()
        self.data.recovery_symbols.return_value = ()
        self.strategy = trading.AUTOBN(self.data)
        self.sample = [99.0, 101.0] * 12
        self.expected = trading.sample_probability(self.sample, 200.0)
        self.expected = self.expected["mean"] + 10 * self.expected["std"]
        self.data.oi_history = AsyncMock(return_value=[
            {"timestamp": i, "sumOpenInterest": value}
            for i, value in enumerate(self.sample + [150.0] * 6)
        ])
        self.data.oi_5m = AsyncMock(return_value=[{"sumOpenInterest": 200.0}])
        self.data.lsr_1h = AsyncMock(return_value=[
            {"timestamp": i, "longAccount": 0.5} for i in range(30)
        ])
        self.data.lsr_5m = AsyncMock(return_value=[
            {"longShortRatio": 0.5, "longAccount": 0.3}
        ])

    async def test_long_entry_saves_chebyshev_threshold_below_90_percent(self):
        self.data.basis_rate = AsyncMock(return_value=0.0)
        bars = replace(make_bars(close=101.0), highs=(103.0,) * 30, lows=(97.0,) * 30)
        observation = make_observation(
            side=trading.OrderSide.BUY, strategy=(trading.StrategyTag.BZ,),
            earliest_open_timestamp=None,
        )
        decision = await self.strategy.evaluate_signal(
            SYMBOL, bars, trading.StateSnapshot({}, {SYMBOL: observation}),
            trading.MarketContext(now=NOW),
        )
        self.assertIsNotNone(decision.intent)
        self.assertAlmostEqual(decision.intent.position.guard.open_interest, self.expected)
        self.assertLess(self.expected, 200.0 * 0.9)

    async def test_recovery_ignores_current_and_historical_90_percent_floors(self):
        for latest, recent in ((200.0, 150.0), (120.0, 300.0)):
            with self.subTest(latest=latest, recent=recent):
                self.data.oi_5m.return_value = [{"sumOpenInterest": latest}]
                self.data.oi_history.return_value = [
                    {"sumOpenInterest": value} for value in self.sample + [recent] * 6
                ]
                current, threshold = await self.strategy._recovery_oi_threshold(SYMBOL, NOW)
                self.assertEqual(current, latest)
                self.assertAlmostEqual(threshold, self.expected)

    async def test_short_entry_keeps_90_percent_drawdown_without_guard(self):
        for current, passed in ((90.0, True), (90.1, False)):
            with self.subTest(current=current):
                self.data.bd_oi_windows = AsyncMock(return_value=((current,), (100.0,)))
                self.assertEqual(
                    await self.strategy.check_side(SYMBOL, "SHORT", NOW),
                    (passed, None, None),
                )
        self.data.oi_history.assert_not_awaited()
        self.data.lsr_1h.assert_not_awaited()


class AUTOBNInitialPriceStopTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_price_stop_before_and_after_take_profit(self):
        for count, price, stop, reason in (
            (0, 90.0, 90.0, "初始止损"),
            (0, 89.0, 90.0, "初始止损"),
            (0, 91.0, 90.0, None),
            (1, 90.0, 90.0, "止盈后价格止损"),
            (0, 90.0, 0.0, None),
        ):
            with self.subTest(count=count, price=price, stop=stop):
                data = Mock()
                data.recovery_symbols.return_value = ()
                data.oi_5m = AsyncMock(return_value=[{"sumOpenInterest": 500.0}])
                data.oi_history = AsyncMock(return_value=None)
                strategy = trading.AUTOBN(data)
                held = trading.Position(
                    take_profit=200.0, stop_loss=stop,
                    position_side=trading.PositionSide.LONG, entry_price=100.0,
                    name=SYMBOL, date=0, strategy=(trading.StrategyTag.BZ,),
                    tp_count=count, guard=trading.OIStop(open_interest=100.0),
                )
                with patch.object(strategy, "calculate_atr", return_value=0.0):
                    decision = await strategy.manage_position(
                        SYMBOL, make_bars(close=price),
                        trading.StateSnapshot({SYMBOL: held}, {}),
                        trading.MarketContext(now=NOW),
                    )
                if reason:
                    self.assertEqual(decision.intent.kind, trading.ActionKind.CLOSE)
                    self.assertEqual(decision.intent.position.close_reason, reason)
                    self.assertEqual(decision.intent.ratio, 1.0)
                    data.oi_5m.assert_not_awaited()
                else:
                    self.assertTrue(decision.intent is None or decision.intent.kind != trading.ActionKind.CLOSE)


class AUTOBNAddBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_uses_entry_regardless_of_hl2_and_strict_boundary(self):
        await self.check_cases(trading.PositionSide.LONG,
                               ((90.0, 90.0, True), (90.0, 85.0, True),
                                (90.0, 84.0, True), (110.0, 95.0, False),
                                (110.0, 94.0, True)))

    async def test_short_uses_entry_regardless_of_hl2_and_strict_boundary(self):
        await self.check_cases(trading.PositionSide.SHORT,
                               ((110.0, 110.0, True), (110.0, 115.0, True),
                                (110.0, 116.0, True), (90.0, 105.0, False),
                                (90.0, 106.0, True)))

    async def check_cases(self, side, cases):
        data = Mock()
        data.recovery_symbols.return_value = ()
        data.oi_5m = AsyncMock(return_value=[{"sumOpenInterest": 500.0}])
        strategy = trading.AUTOBN(data)
        held = trading.Position(
            take_profit=200.0 if side == trading.PositionSide.LONG else 10.0,
            stop_loss=10.0 if side == trading.PositionSide.LONG else 200.0,
            position_side=side, entry_price=100.0, name=SYMBOL, date=0,
            strategy=(trading.StrategyTag.BZ if side == trading.PositionSide.LONG
                      else trading.StrategyTag.BD,),
            guard=trading.OIStop(open_interest=100.0),
        )
        for midprice, price, should_add in cases:
            with self.subTest(side=side, midprice=midprice, price=price):
                bars = replace(make_bars(close=price),
                               highs=(midprice + 30.0,) * 30,
                               lows=(midprice - 30.0,) * 30)
                self.assertEqual(bars.midprice, midprice)
                with patch.object(strategy, "calculate_atr", return_value=5.0):
                    decision = await strategy.manage_position(
                        SYMBOL, bars, trading.StateSnapshot({SYMBOL: held}, {}),
                        trading.MarketContext(now=NOW),
                    )
                if should_add:
                    self.assertEqual(decision.intent.kind, trading.ActionKind.ADD)
                    self.assertEqual(decision.intent.position, held)
                else:
                    self.assertIsNone(decision.intent)


class AUTOBNPostProfitOITrackingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.data = Mock()
        self.data.recovery_symbols.return_value = ()
        self.data.oi_history = AsyncMock(return_value=[{"sumOpenInterest": 200.0}] * 30)
        self.data.oi_5m = AsyncMock(return_value=[{"sumOpenInterest": 500.0}])
        self.strategy = trading.AUTOBN(self.data)
        self.held = trading.Position(
            take_profit=200.0, stop_loss=50.0,
            position_side=trading.PositionSide.LONG, entry_price=100.0,
            name=SYMBOL, date=0, strategy=(trading.StrategyTag.BZ,),
            tp_count=1, guard=trading.OIStop(open_interest=100.0),
        )

    async def check(self, held=None, price=100.0):
        return await self.strategy.manage_position(
            SYMBOL, replace(make_bars(close=price), highs=(101.0,) * 30, lows=(99.0,) * 30),
            trading.StateSnapshot({SYMBOL: held or self.held}, {}),
            trading.MarketContext(now=NOW),
        )

    async def test_each_check_raises_guard_without_another_take_profit(self):
        held = self.held
        for peak, expected in ((200.0, 180.0), (300.0, 270.0), (150.0, 270.0)):
            self.data.oi_history.return_value = [{"sumOpenInterest": 1.0}] * 29 + [{"sumOpenInterest": peak}]
            decision = await self.check(held)
            self.assertIsNone(decision.intent)
            held = dict(decision.patch.positions)[SYMBOL]
            self.assertEqual(held.guard.open_interest, expected)
            self.assertEqual(held.tp_count, 1)
        self.assertEqual(self.data.oi_history.await_count, 3)

    async def test_new_guard_triggers_oi_stop_in_same_check(self):
        self.data.oi_5m.return_value = [{"sumOpenInterest": 180.0}]
        decision = await self.check()
        self.assertEqual(decision.intent.position.close_reason, "OI止损")
        self.assertEqual(decision.intent.position.guard.open_interest, 180.0)
        self.assertEqual(dict(decision.patch.positions)[SYMBOL].guard.open_interest, 180.0)

    async def test_exit_and_add_intents_keep_updated_guard(self):
        for price, kind, reason in ((40.0, trading.ActionKind.CLOSE, "止盈后价格止损"),
                                     (210.0, trading.ActionKind.CLOSE, "止盈"),
                                     (90.0, trading.ActionKind.ADD, "")):
            with self.subTest(price=price):
                decision = await self.check(price=price)
                self.assertEqual(decision.intent.kind, kind)
                self.assertEqual(decision.intent.position.close_reason, reason)
                self.assertEqual(decision.intent.position.guard.open_interest, 180.0)
                self.assertEqual(dict(decision.patch.positions)[SYMBOL].guard.open_interest, 180.0)

    async def test_no_history_query_before_profit_or_for_short(self):
        for updates in ({"tp_count": 0}, {"position_side": trading.PositionSide.SHORT}):
            await self.check(self.held.model_copy(update=updates))
        self.data.oi_history.assert_not_awaited()

    async def test_bad_history_preserves_old_guard_and_exit(self):
        for history in (None, [], [{"sumOpenInterest": 200.0}] * 29,
                        [{"sumOpenInterest": value} for value in [200.0] * 29 + [float("nan")]],
                        [{"sumOpenInterest": 0.0}] * 30):
            with self.subTest(history=history):
                self.data.oi_history.return_value = history
                self.data.oi_5m.return_value = [{"sumOpenInterest": 100.0}]
                decision = await self.check()
                self.assertEqual(decision.intent.position.close_reason, "OI止损")
                self.assertEqual(decision.intent.position.guard.open_interest, 100.0)
        self.data.oi_history.side_effect = RuntimeError("offline")
        for price, reason in ((40.0, "止盈后价格止损"), (210.0, "止盈"), (100.0, "OI止损")):
            decision = await self.check(price=price)
            self.assertEqual(decision.intent.position.close_reason, reason)
            self.assertEqual(decision.intent.position.guard.open_interest, 100.0)

    async def test_real_data_rejects_stale_fallback_but_accepts_current_window(self):
        boundary = int(NOW.timestamp() * 1000)
        for response in (RuntimeError("offline"), [], [{"timestamp": boundary, "sumOpenInterest": 200.0}], "fresh", "cached"):
            with self.subTest(response=response):
                gateway = Mock()
                data = trading.BinanceMarketData(gateway, Mock())
                current = [{"timestamp": boundary - (29 - i) * 3600000,
                            "sumOpenInterest": 200.0} for i in range(30)]
                stale = [dict(row, timestamp=row["timestamp"] - 3600000) for row in current]
                data.cache.oi_history[(SYMBOL, "1h")] = current if response == "cached" else stale
                gateway.call = AsyncMock()
                if isinstance(response, Exception):
                    gateway.call.side_effect = response
                else:
                    gateway.call.return_value = current if response == "fresh" else response
                data.oi_5m = AsyncMock(return_value=[{"sumOpenInterest": 150.0}])
                data.recovery_symbols = Mock(return_value=())
                self.strategy = trading.AUTOBN(data)
                decision = await self.check()
                if response in ("fresh", "cached"):
                    self.assertEqual(decision.intent.position.close_reason, "OI止损")
                    self.assertEqual(decision.intent.position.guard.open_interest, 180.0)
                else:
                    self.assertIsNone(decision.intent)
                    self.assertEqual(dict(decision.patch.positions)[SYMBOL].guard.open_interest, 100.0)
                if response == "cached":
                    gateway.call.assert_not_awaited()

    async def test_history_cancellation_propagates(self):
        self.data.oi_history.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.check()


class AUTOBNBDObservationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.data = Mock()
        self.data.recovery_symbols.return_value = ()
        self.data.bd_oi_windows = AsyncMock(return_value=(OI_WINDOW, OI_WINDOW[:-1]))
        self.data.gateway.diagnostics.note = Mock()
        self.strategy = trading.AUTOBN(self.data)
        self.context = trading.MarketContext(now=NOW)

    async def call_maintain(self, bars, *, observation=None, positions=None):
        observations = {SYMBOL: observation} if observation is not None else {}
        state = trading.StateSnapshot(positions or {}, observations)
        return await self.strategy.maintain_observation(SYMBOL, bars, state, self.context)

    @patch.object(
        trading,
        "sample_probability",
        return_value={"chebyshev_upper_bound": 0.01},
    )
    async def test_initial_bd_is_created_with_one_oi_window(self, _probability):
        result = await self.call_maintain(make_bars())

        self.data.bd_oi_windows.assert_awaited_once_with(SYMBOL, NOW)
        self.assertEqual(len(result.observations), 1)
        symbol, observation = result.observations[0]
        self.assertEqual(symbol, SYMBOL)
        self.assertEqual(observation.strategy, (trading.StrategyTag.BD,))
        self.assertEqual(observation.side, trading.OrderSide.SELL)
        self.assertEqual(observation.price, 100.0)
        self.assertEqual(observation.timestamp, NOW.timestamp())

    async def test_refresh_only_preserves_existing_bd_fields(self):
        existing = make_observation(
            bz_reference_high=123.0,
            reopen_pending_date="2026-09-16",
        )
        result = await self.call_maintain(
            make_bars(peak_index=10), observation=existing
        )

        self.data.bd_oi_windows.assert_awaited_once_with(SYMBOL, NOW)
        self.assertEqual(result.observations[0][0], SYMBOL)
        refreshed = result.observations[0][1]
        self.assertEqual(refreshed.price, 100.0)
        self.assertEqual(refreshed.timestamp, NOW.timestamp())
        self.assertEqual(refreshed.bz_reference_high, 123.0)
        self.assertEqual(refreshed.reopen_pending_date, "2026-09-16")
        self.assertEqual(
            refreshed.earliest_open_timestamp, existing.earliest_open_timestamp
        )

    @patch.object(
        trading,
        "sample_probability",
        return_value={"chebyshev_upper_bound": 0.01},
    )
    async def test_existing_bd_that_meets_initial_rule_is_rebuilt(self, _probability):
        existing = make_observation(
            bz_reference_high=123.0,
            reopen_pending_date="2026-09-16",
        )
        result = await self.call_maintain(make_bars(), observation=existing)

        rebuilt = result.observations[0][1]
        self.assertEqual(
            rebuilt.earliest_open_timestamp, existing.earliest_open_timestamp
        )
        self.assertIsNone(rebuilt.bz_reference_high)
        self.assertIsNone(rebuilt.reopen_pending_date)

    async def test_new_volume_peak_deletes_existing_bd_without_oi_request(self):
        result = await self.call_maintain(
            make_bars(close=100.0, peak_index=10, last_volume=100.0),
            observation=make_observation(),
        )

        self.assertEqual(result.observations, ((SYMBOL, None),))
        self.data.bd_oi_windows.assert_not_awaited()

    async def test_bz_after_bd_deletion_does_not_inherit_cooldown(self):
        existing = make_observation()
        result = await self.call_maintain(
            make_bars(close=101.0, peak_index=10, last_volume=100.0),
            observation=existing,
        )

        replacement = result.observations[0][1]
        self.assertEqual(replacement.strategy, (trading.StrategyTag.BZ,))
        self.assertIsNone(replacement.earliest_open_timestamp)
        self.data.bd_oi_windows.assert_not_awaited()

    async def test_bz_wins_without_requesting_bd_oi_or_inheriting_cooldown(self):
        existing = make_observation()
        result = await self.call_maintain(
            make_bars(close=101.0, peak_index=10, last_volume=20.0),
            observation=existing,
        )

        replacement = result.observations[0][1]
        self.assertEqual(replacement.strategy, (trading.StrategyTag.BZ,))
        self.assertIsNone(replacement.earliest_open_timestamp)
        self.data.bd_oi_windows.assert_not_awaited()

    async def test_same_direction_bz_rebuild_keeps_cooldown(self):
        existing = make_observation(
            side=trading.OrderSide.BUY,
            strategy=(trading.StrategyTag.BZ,),
        )
        result = await self.call_maintain(
            make_bars(close=101.0, peak_index=10, last_volume=20.0),
            observation=existing,
        )

        replacement = result.observations[0][1]
        self.assertEqual(replacement.strategy, (trading.StrategyTag.BZ,))
        self.assertEqual(
            replacement.earliest_open_timestamp,
            existing.earliest_open_timestamp,
        )
        self.data.bd_oi_windows.assert_not_awaited()

    @patch.object(
        trading,
        "sample_probability",
        return_value={"chebyshev_upper_bound": 0.01},
    )
    async def test_bd_replacing_bz_does_not_inherit_cooldown(self, _probability):
        existing = make_observation(
            side=trading.OrderSide.BUY,
            strategy=(trading.StrategyTag.BZ,),
        )
        result = await self.call_maintain(make_bars(), observation=existing)

        replacement = result.observations[0][1]
        self.assertEqual(replacement.strategy, (trading.StrategyTag.BD,))
        self.assertIsNone(replacement.earliest_open_timestamp)
        self.data.bd_oi_windows.assert_awaited_once_with(SYMBOL, NOW)

    async def test_refresh_evaluates_bd_prefilter_once(self):
        await self.call_maintain(
            make_bars(peak_index=10), observation=make_observation()
        )

        self.data.gateway.diagnostics.note.assert_called_once_with(
            "prefilter", "bd", "", "passed"
        )

    @patch.object(
        trading,
        "sample_probability",
        return_value={"chebyshev_upper_bound": 0.01},
    )
    async def test_held_symbol_keeps_current_initial_bd_behavior(self, _probability):
        held = object()
        result = await self.call_maintain(
            make_bars(), positions={SYMBOL: held}
        )

        self.assertEqual(
            result.observations[0][1].strategy, (trading.StrategyTag.BD,)
        )
        self.data.bd_oi_windows.assert_awaited_once_with(SYMBOL, NOW)

    async def test_held_bd_refreshes_and_new_peak_deletes_before_signal(self):
        existing = make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None)
        held = object()
        refreshed = await self.call_maintain(
            make_bars(peak_index=10), observation=existing, positions={SYMBOL: held}
        )
        self.assertEqual(refreshed.observations[0][1].timestamp, NOW.timestamp())
        self.assertEqual(refreshed.observations[0][1].price, 100.0)
        removed = await self.call_maintain(
            make_bars(close=99.0, peak_index=10, last_volume=100.0),
            observation=existing, positions={SYMBOL: held},
        )
        self.assertEqual(removed.observations, ((SYMBOL, None),))


class AUTOBNCooldownEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.state = trading.TradingState("AUTOBN", f"{directory.name}/state.json")
        self.data = Mock()
        self.data.recovery_symbols.return_value = ()
        self.data.wait_ready = AsyncMock(return_value=True)
        self.data.fetch_bars = AsyncMock()
        self.data.bd_oi_windows = AsyncMock(return_value=(OI_WINDOW, OI_WINDOW[:-1]))
        self.data.oi_5m = AsyncMock(return_value=[{"sumOpenInterest": 200.0}])
        self.data.basis_rate = AsyncMock(return_value=0.0)
        self.strategy = trading.AUTOBN(self.data)
        self.data.lsr_1h = AsyncMock(return_value=[])
        self.data.lsr_5m = AsyncMock(return_value=[])
        self.data.oi_history = AsyncMock(return_value=[])
        self.executor = Mock()
        self.executor.REQUIRES_ORDER_JOURNAL = False
        self.executor.prepare = AsyncMock(
            side_effect=lambda intent: trading.PreparedOrder(
                intent=intent,
                quantity=1.0,
                price=intent.price,
                entry_price=intent.position.entry_price,
                account_equity=1000.0,
                notional=intent.price,
            )
        )
        self.executor.submit = AsyncMock(
            side_effect=lambda order: trading.ExecutionResult(
                status=trading.ExecutionStatus.FILLED,
                quantity=1.0,
                price=order.price,
                entry_price=order.entry_price,
                ratio=1.0,
                original_quantity=1.0,
                remaining_quantity=0
                if order.intent.kind == trading.ActionKind.CLOSE
                else 1.0,
            )
        )
        self.executor.complete_result = AsyncMock(
            side_effect=lambda order, result: result
        )
        notifications = Mock()
        notifications.send = AsyncMock()
        self.engine = trading.TradingEngine(
            self.strategy,
            self.executor,
            self.state,
            notifications,
            Mock(),
            Mock(),
            Mock(),
        )

    async def scan(self, bars, now=NOW):
        # 非零振幅让开仓/持仓管理使用真实 ATR 计算。
        self.data.fetch_bars.return_value = replace(
            bars,
            highs=tuple(max(102.0, price + 2) for price in bars.closes),
            lows=tuple(min(98.0, price - 2) for price in bars.closes),
        )
        outcome = await self.engine.process_instrument(
            SYMBOL, trading.MarketContext(now=now)
        )
        self.assertEqual(outcome, trading.ScanOutcome.PROCESSED)
        return self.state.snapshot()

    def seed(self, observation, position=None):
        self.state.apply(
            trading.StatePatch(
                observations=((SYMBOL, observation),),
                positions=((SYMBOL, position),),
            )
        )

    def cooling_bz(self, **updates):
        return make_observation(
            side=trading.OrderSide.BUY,
            strategy=(trading.StrategyTag.BZ,),
            **{"timestamp": NOW.timestamp() - 3600, **updates},
        )

    def position(self, side):
        is_long = side == trading.PositionSide.LONG
        return trading.Position(
            take_profit=130.0 if is_long else 70.0,
            stop_loss=80.0 if is_long else 110.0,
            position_side=side,
            entry_price=100.0,
            name=SYMBOL,
            date=20260916,
            strategy=(trading.StrategyTag.BZ if is_long else trading.StrategyTag.BD,),
            guard=trading.OIStop(open_interest=150.0 if is_long else 0.0),
        )

    async def test_cooling_bz_switches_to_bd_then_can_open_next_scan(self):
        self.seed(self.cooling_bz())
        snapshot = await self.scan(make_bars())

        self.assertEqual(
            snapshot.observations[SYMBOL].strategy, (trading.StrategyTag.BD,)
        )
        self.assertIsNone(snapshot.observations[SYMBOL].earliest_open_timestamp)
        self.assertNotIn(SYMBOL, snapshot.positions)
        self.executor.prepare.assert_not_awaited()

        self.data.bd_oi_windows.return_value = (
            OI_WINDOW[:-1] + (80.0,),
            OI_WINDOW[:-1],
        )
        snapshot = await self.scan(
            make_bars(close=99.0), NOW + datetime.timedelta(minutes=5)
        )
        self.assertEqual(
            snapshot.positions[SYMBOL].position_side, trading.PositionSide.SHORT
        )
        self.executor.submit.assert_awaited_once()

    async def test_cooling_bd_switches_to_bz_without_opening_same_scan(self):
        self.seed(make_observation(timestamp=NOW.timestamp()))
        snapshot = await self.scan(
            make_bars(close=101.0, peak_index=10, last_volume=20.0)
        )

        self.assertEqual(
            snapshot.observations[SYMBOL].strategy, (trading.StrategyTag.BZ,)
        )
        self.assertIsNone(snapshot.observations[SYMBOL].earliest_open_timestamp)
        self.executor.prepare.assert_not_awaited()
        self.data.bd_oi_windows.assert_not_awaited()

    async def test_cooling_same_side_refreshes_without_entry_confirmation(self):
        observation = self.cooling_bz()
        self.seed(observation)
        with patch.object(
            self.strategy, "check_side", new_callable=AsyncMock
        ) as check_side:
            snapshot = await self.scan(
                make_bars(close=101.0, peak_index=10, last_volume=20.0)
            )

        refreshed = snapshot.observations[SYMBOL]
        self.assertEqual(refreshed.timestamp, NOW.timestamp())
        self.assertEqual(
            refreshed.earliest_open_timestamp, observation.earliest_open_timestamp
        )
        check_side.assert_not_awaited()
        self.executor.prepare.assert_not_awaited()

    async def test_expired_observation_keeps_active_cooldown_across_scans(self):
        observation = self.cooling_bz(timestamp=NOW.timestamp() - 2 * 86400)
        self.seed(observation)
        with patch.object(
            self.strategy, "check_side", new_callable=AsyncMock
        ) as check_side:
            # 无新观察时也保留冷却，随后同向重建仍不能开仓。
            snapshot = await self.scan(make_bars(close=99.0))
            self.assertEqual(snapshot.observations[SYMBOL], observation)
            for minutes in (5, 10):
                snapshot = await self.scan(
                    make_bars(close=101.0, peak_index=10, last_volume=20.0),
                    NOW + datetime.timedelta(minutes=minutes),
                )
                self.assertEqual(
                    snapshot.observations[SYMBOL].earliest_open_timestamp,
                    observation.earliest_open_timestamp,
                )
        check_side.assert_not_awaited()
        self.executor.prepare.assert_not_awaited()

    async def test_cooldown_expiry_allows_entry_confirmation(self):
        self.seed(self.cooling_bz(earliest_open_timestamp=NOW.timestamp()))
        with patch.object(
            self.strategy, "check_side", new=AsyncMock(return_value=(True, 1.0, 150.0))
        ) as check_side:
            snapshot = await self.scan(
                make_bars(close=101.0, peak_index=10, last_volume=20.0)
            )

        check_side.assert_awaited_once_with(
            SYMBOL, trading.PositionSide.LONG.value, NOW
        )
        self.assertEqual(
            snapshot.positions[SYMBOL].position_side, trading.PositionSide.LONG
        )
        self.executor.submit.assert_awaited_once()

    async def test_expired_observation_is_removed_after_cooldown_ends(self):
        self.seed(
            self.cooling_bz(
                timestamp=NOW.timestamp() - 2 * 86400,
                earliest_open_timestamp=NOW.timestamp(),
            )
        )
        snapshot = await self.scan(make_bars(close=99.0))
        self.assertNotIn(SYMBOL, snapshot.observations)
        self.executor.prepare.assert_not_awaited()

    async def test_long_close_opposite_bd_opens_same_scan_without_cooldown(self):
        self.seed(
            self.cooling_bz(earliest_open_timestamp=None),
            self.position(trading.PositionSide.LONG),
        )
        snapshot = await self.scan(make_bars())
        self.assertEqual(snapshot.observations[SYMBOL].strategy, (trading.StrategyTag.BD,))
        self.assertEqual(snapshot.positions[SYMBOL].position_side, trading.PositionSide.LONG)

        self.data.bd_oi_windows.return_value = (OI_WINDOW[:-1] + (80.0,), OI_WINDOW[:-1])
        with patch.object(self.strategy, "check_side", new=AsyncMock(return_value=(True, 1.0, 150.0))):
            snapshot = await self.scan(make_bars(close=99.0), NOW + datetime.timedelta(minutes=5))
        self.assertEqual(snapshot.positions[SYMBOL].position_side, trading.PositionSide.SHORT)
        self.assertIsNone(snapshot.observations[SYMBOL].earliest_open_timestamp)
        self.assertEqual(
            [call.args[0].intent.kind for call in self.executor.submit.await_args_list],
            [trading.ActionKind.CLOSE, trading.ActionKind.OPEN],
        )

    async def test_short_close_preserves_preexisting_bz(self):
        self.seed(
            make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
            self.position(trading.PositionSide.SHORT),
        )
        snapshot = await self.scan(
            make_bars(close=101.0, peak_index=10, last_volume=20.0)
        )
        bz = snapshot.observations[SYMBOL]
        self.assertIn(SYMBOL, snapshot.positions)
        self.assertEqual(bz.strategy, (trading.StrategyTag.BZ,))
        self.executor.prepare.assert_not_awaited()

        self.data.bd_oi_windows.return_value = (None, None)
        snapshot = await self.scan(
            make_bars(close=111.0), NOW + datetime.timedelta(minutes=1)
        )
        self.assertNotIn(SYMBOL, snapshot.positions)
        self.assertEqual(snapshot.observations[SYMBOL], bz)

    async def test_same_side_long_close_keeps_24_hour_cooldown(self):
        self.seed(
            self.cooling_bz(earliest_open_timestamp=None),
            self.position(trading.PositionSide.LONG),
        )
        self.data.oi_5m.return_value = [{"sumOpenInterest": 80.0}]
        snapshot = await self.scan(make_bars(close=99.0))
        self.assertNotIn(SYMBOL, snapshot.positions)
        self.assertEqual(
            snapshot.observations[SYMBOL].earliest_open_timestamp,
            NOW.timestamp() + self.strategy.REOPEN_COOLDOWN_SECONDS,
        )

    async def test_same_side_short_close_deletes_bd(self):
        self.seed(
            make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
            self.position(trading.PositionSide.SHORT),
        )
        self.data.bd_oi_windows.return_value = (None, None)
        snapshot = await self.scan(make_bars(close=111.0))
        self.assertNotIn(SYMBOL, snapshot.positions)
        self.assertNotIn(SYMBOL, snapshot.observations)


class AUTOBNSameCycleEntryTests(unittest.IsolatedAsyncioTestCase):
    setUp = AUTOBNCooldownEngineTests.setUp
    scan = AUTOBNCooldownEngineTests.scan
    seed = AUTOBNCooldownEngineTests.seed
    position = AUTOBNCooldownEngineTests.position

    async def test_new_bz_observation_opens_in_same_scan(self):
        bars = make_bars(close=101.0, peak_index=10, last_volume=20.0)
        with patch.object(self.strategy, "check_side", new=AsyncMock(return_value=(True, 1.0, 150.0))) as check:
            snapshot = await self.scan(bars)
        self.assertEqual(snapshot.positions[SYMBOL].position_side, trading.PositionSide.LONG)
        check.assert_awaited_once()
        self.data.bd_oi_windows.assert_not_awaited()

    async def test_reversal_closes_before_open_and_confirms_once(self):
        for held_side, observation, bars, target in (
            (trading.PositionSide.SHORT, make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
             make_bars(close=101.0, peak_index=10, last_volume=20.0), trading.PositionSide.LONG),
            (trading.PositionSide.LONG, make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
             make_bars(close=99.0, peak_index=10), trading.PositionSide.SHORT),
        ):
            with self.subTest(side=held_side):
                self.executor.submit.reset_mock()
                self.executor.prepare.reset_mock()
                self.data.bd_oi_windows.reset_mock()
                self.seed(observation, self.position(held_side))
                with patch.object(self.strategy, "check_side", new=AsyncMock(return_value=(True, 1.0, 150.0))) as check, patch.object(
                    self.strategy, "maintain_observation", wraps=self.strategy.maintain_observation
                ) as maintain:
                    snapshot = await self.scan(bars)
                self.assertEqual(snapshot.positions[SYMBOL].position_side, target)
                self.assertEqual([call.args[0].intent.kind for call in self.executor.submit.await_args_list],
                                 [trading.ActionKind.CLOSE, trading.ActionKind.OPEN])
                check.assert_awaited_once()
                maintain.assert_awaited_once()

    async def test_deleted_bd_cannot_reverse_and_same_side_skips_confirmation(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        with patch.object(self.strategy, "check_side", new_callable=AsyncMock) as check:
            snapshot = await self.scan(make_bars(close=99.0, peak_index=10, last_volume=100.0))
        self.assertNotIn(SYMBOL, snapshot.observations)
        self.assertEqual(snapshot.positions[SYMBOL].position_side, trading.PositionSide.LONG)
        check.assert_not_awaited()
        self.executor.submit.assert_not_awaited()

        self.executor.submit.reset_mock()
        self.seed(make_observation(side=trading.OrderSide.BUY,
                                   strategy=(trading.StrategyTag.BZ,),
                                   timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        with patch.object(self.strategy, "check_side", new_callable=AsyncMock) as check:
            await self.scan(make_bars(close=101.0, peak_index=10, last_volume=20.0))
        check.assert_not_awaited()
        self.executor.submit.assert_not_awaited()

    async def test_normal_exit_does_not_reuse_preclose_observation_for_entry(self):
        self.seed(make_observation(side=trading.OrderSide.BUY,
                                   strategy=(trading.StrategyTag.BZ,),
                                   timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        self.data.oi_5m.return_value = [{"sumOpenInterest": 80.0}]
        with patch.object(self.strategy, "check_side", new_callable=AsyncMock) as check:
            snapshot = await self.scan(make_bars(close=99.0, peak_index=10))
        self.assertNotIn(SYMBOL, snapshot.positions)
        self.assertEqual(len(self.executor.submit.await_args_list), 1)
        check.assert_not_awaited()

    async def test_opposite_observation_without_full_signal_does_not_close(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        with patch.object(self.strategy, "check_side", new=AsyncMock(return_value=(False, None, None))):
            snapshot = await self.scan(make_bars(close=99.0, peak_index=10))
        self.assertIn(SYMBOL, snapshot.positions)
        self.executor.submit.assert_not_awaited()

    async def test_failed_partial_unknown_close_never_opens(self):
        for result in (
            trading.ExecutionResult(trading.ExecutionStatus.FAILED, 0, 0, 100, 1, 1),
            trading.ExecutionResult(trading.ExecutionStatus.UNKNOWN, 0, 0, 100, 1, 1),
            trading.ExecutionResult(trading.ExecutionStatus.FILLED, 0.5, 99, 100, 1, 1, remaining_quantity=0.5),
        ):
            with self.subTest(status=result.status):
                self.executor.submit.reset_mock()
                self.executor.submit.return_value = result
                self.executor.submit.side_effect = None
                self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                          self.position(trading.PositionSide.LONG))
                with patch.object(self.strategy, "check_side", new=AsyncMock(return_value=(True, 1.0, 150.0))):
                    await self.scan(make_bars(close=99.0, peak_index=10))
                self.assertEqual(len(self.executor.submit.await_args_list), 1)
                self.assertEqual(self.executor.submit.await_args.args[0].intent.kind, trading.ActionKind.CLOSE)

    async def test_pending_after_close_blocks_open_even_if_close_reported_success(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        pending = Mock()
        self.state.pending_order = Mock(side_effect=[None, None, pending])
        self.data.fetch_bars.return_value = replace(
            make_bars(close=99.0, peak_index=10),
            highs=tuple([102.0] * 30), lows=tuple([98.0] * 30),
        )
        with patch.object(self.strategy, "check_side", new=AsyncMock(return_value=(True, 1.0, 150.0))):
            outcome = await self.engine.process_instrument(SYMBOL, trading.MarketContext(now=NOW))
        self.assertEqual(outcome, trading.ScanOutcome.PENDING)
        self.assertEqual(len(self.executor.submit.await_args_list), 1)
        self.assertEqual(self.executor.submit.await_args.args[0].intent.kind, trading.ActionKind.CLOSE)

    async def test_open_rejected_after_full_close_remains_flat(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        original_prepare = self.executor.prepare.side_effect
        async def prepare(intent):
            return None if intent.kind == trading.ActionKind.OPEN else original_prepare(intent)
        self.executor.prepare.side_effect = prepare
        with patch.object(self.strategy, "check_side", new=AsyncMock(return_value=(True, 1.0, 150.0))):
            snapshot = await self.scan(make_bars(close=99.0, peak_index=10))
        self.assertNotIn(SYMBOL, snapshot.positions)
        self.assertEqual(len(self.executor.submit.await_args_list), 1)

    async def test_held_bd_maintenance_failure_still_checks_existing_exit(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        self.data.bd_oi_windows.side_effect = RuntimeError("BD OI unavailable")
        self.data.oi_5m.return_value = [{"sumOpenInterest": 80.0}]
        with patch.object(self.strategy, "check_side", new_callable=AsyncMock) as check:
            snapshot = await self.scan(make_bars(close=100.0, peak_index=10))
        self.assertNotIn(SYMBOL, snapshot.positions)
        self.assertEqual(self.executor.submit.await_args.args[0].intent.kind, trading.ActionKind.CLOSE)
        self.assertEqual(self.executor.submit.await_args.args[0].intent.position.close_reason, "OI止损")
        check.assert_not_awaited()

    async def test_held_reversal_confirmation_failure_still_manages_position(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        self.data.bd_oi_windows.return_value = (OI_WINDOW[:-1] + (80.0,), OI_WINDOW[:-1])
        self.data.oi_5m.return_value = [{"sumOpenInterest": 80.0}]
        with patch.object(self.strategy, "check_side", new=AsyncMock(side_effect=RuntimeError("entry OI unavailable"))) as check:
            snapshot = await self.scan(make_bars(close=99.0, peak_index=10))
        self.assertNotIn(SYMBOL, snapshot.positions)
        self.assertEqual(self.executor.submit.await_args.args[0].intent.position.close_reason, "OI止损")
        check.assert_awaited_once()
        self.assertEqual(len(self.executor.submit.await_args_list), 1)

    async def test_flat_observation_failure_never_opens(self):
        self.data.bd_oi_windows.side_effect = RuntimeError("BD OI unavailable")
        with patch.object(self.strategy, "check_side", new_callable=AsyncMock) as check:
            self.data.fetch_bars.return_value = make_bars(close=100.0)
            outcome = await self.engine.process_instrument(SYMBOL, trading.MarketContext(now=NOW))
        self.assertEqual(outcome, trading.ScanOutcome.FAILED)
        self.executor.prepare.assert_not_awaited()
        check.assert_not_awaited()

    async def test_no_position_close_stays_flat_without_reversal_open(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        self.executor.submit.side_effect = lambda order: trading.ExecutionResult(
            trading.ExecutionStatus.NO_POSITION, 0, order.price, order.entry_price, 1, 1
        )
        with patch.object(self.strategy, "check_side", new=AsyncMock(return_value=(True, 1.0, 150.0))):
            snapshot = await self.scan(make_bars(close=99.0, peak_index=10))
        self.assertNotIn(SYMBOL, snapshot.positions)
        self.assertEqual(len(self.executor.submit.await_args_list), 1)
        self.assertEqual(self.executor.submit.await_args.args[0].intent.kind, trading.ActionKind.CLOSE)

    async def test_close_completion_exception_never_opens(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        self.executor.complete_result.side_effect = RuntimeError("close query failed")
        self.data.fetch_bars.return_value = replace(
            make_bars(close=99.0, peak_index=10),
            highs=tuple([102.0] * 30), lows=tuple([98.0] * 30),
        )
        with patch.object(self.strategy, "check_side", new=AsyncMock(return_value=(True, 1.0, 150.0))):
            outcome = await self.engine.process_instrument(SYMBOL, trading.MarketContext(now=NOW))
        self.assertEqual(outcome, trading.ScanOutcome.FAILED)
        self.assertEqual(len(self.executor.submit.await_args_list), 1)

    async def test_cancelled_observation_propagates_without_orders(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        self.data.fetch_bars.return_value = make_bars(close=100.0, peak_index=10)
        with patch.object(self.strategy, "maintain_observation", new=AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                await self.engine.process_instrument(SYMBOL, trading.MarketContext(now=NOW))
        self.executor.prepare.assert_not_awaited()

    async def test_cancelled_close_completion_never_opens(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        self.executor.complete_result.side_effect = asyncio.CancelledError
        self.data.fetch_bars.return_value = replace(
            make_bars(close=99.0, peak_index=10),
            highs=tuple([102.0] * 30), lows=tuple([98.0] * 30),
        )
        with patch.object(self.strategy, "check_side", new=AsyncMock(return_value=(True, 1.0, 150.0))):
            with self.assertRaises(asyncio.CancelledError):
                await self.engine.process_instrument(SYMBOL, trading.MarketContext(now=NOW))
        self.assertEqual(len(self.executor.submit.await_args_list), 1)
        self.assertEqual(self.executor.submit.await_args.args[0].intent.kind, trading.ActionKind.CLOSE)

    async def test_cancelled_reversal_confirmation_propagates_without_orders(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        self.data.fetch_bars.return_value = make_bars(close=99.0, peak_index=10)
        with patch.object(self.strategy, "check_side", new=AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                await self.engine.process_instrument(SYMBOL, trading.MarketContext(now=NOW))
        self.executor.submit.assert_not_awaited()

    async def test_pending_skips_observation_and_signal(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        pending = Mock()
        self.state.pending_order = Mock(return_value=pending)
        self.executor.reconcile = AsyncMock(return_value=Mock(status=trading.ExecutionStatus.UNKNOWN))
        with patch.object(self.strategy, "maintain_observation", new_callable=AsyncMock) as maintain, patch.object(
            self.strategy, "check_side", new_callable=AsyncMock
        ) as check:
            outcome = await self.engine.process_instrument(SYMBOL, trading.MarketContext(now=NOW))
        self.assertEqual(outcome, trading.ScanOutcome.PENDING)
        maintain.assert_not_awaited()
        check.assert_not_awaited()
        self.executor.submit.assert_not_awaited()

    async def test_recovery_is_not_treated_as_reversal(self):
        self.seed(make_observation(timestamp=NOW.timestamp(), earliest_open_timestamp=None),
                  self.position(trading.PositionSide.LONG))
        self.data.recovery_symbols.return_value = (SYMBOL,)
        with patch.object(self.strategy, "check_side", new_callable=AsyncMock) as check:
            await self.scan(make_bars(close=99.0, peak_index=10))
        check.assert_not_awaited()
        self.executor.submit.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
