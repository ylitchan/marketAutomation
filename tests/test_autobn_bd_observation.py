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


class AUTOBNBDObservationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.data = Mock()
        self.data.recovery_symbols.return_value = ()
        self.data.bd_oi_windows = AsyncMock(return_value=(OI_WINDOW, OI_WINDOW[:-1]))
        self.data.gateway.diagnostics.note = Mock()
        self.strategy = trading.AUTOBN(self.data)
        self.context = trading.MarketContext(now=NOW)

    async def call_after(self, bars, *, observation=None, prior_positions=None, positions=None):
        observations = {SYMBOL: observation} if observation is not None else {}
        state = trading.StateSnapshot(positions or {}, observations)
        prior = trading.StateSnapshot(prior_positions or {}, observations)
        return await self.strategy.after_instrument(
            SYMBOL, bars, state, prior, self.context
        )

    @patch.object(
        trading,
        "sample_probability",
        return_value={"chebyshev_upper_bound": 0.01},
    )
    async def test_initial_bd_is_created_with_one_oi_window(self, _probability):
        result = await self.call_after(make_bars())

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
        result = await self.call_after(
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
        result = await self.call_after(make_bars(), observation=existing)

        rebuilt = result.observations[0][1]
        self.assertEqual(
            rebuilt.earliest_open_timestamp, existing.earliest_open_timestamp
        )
        self.assertIsNone(rebuilt.bz_reference_high)
        self.assertIsNone(rebuilt.reopen_pending_date)

    async def test_new_volume_peak_deletes_existing_bd_without_oi_request(self):
        result = await self.call_after(
            make_bars(close=100.0, peak_index=10, last_volume=100.0),
            observation=make_observation(),
        )

        self.assertEqual(result.observations, ((SYMBOL, None),))
        self.data.bd_oi_windows.assert_not_awaited()

    async def test_bz_after_bd_deletion_does_not_inherit_cooldown(self):
        existing = make_observation()
        result = await self.call_after(
            make_bars(close=101.0, peak_index=10, last_volume=100.0),
            observation=existing,
        )

        replacement = result.observations[0][1]
        self.assertEqual(replacement.strategy, (trading.StrategyTag.BZ,))
        self.assertIsNone(replacement.earliest_open_timestamp)
        self.data.bd_oi_windows.assert_not_awaited()

    async def test_bz_wins_without_requesting_bd_oi_or_inheriting_cooldown(self):
        existing = make_observation()
        result = await self.call_after(
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
        result = await self.call_after(
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
        result = await self.call_after(make_bars(), observation=existing)

        replacement = result.observations[0][1]
        self.assertEqual(replacement.strategy, (trading.StrategyTag.BD,))
        self.assertIsNone(replacement.earliest_open_timestamp)
        self.data.bd_oi_windows.assert_awaited_once_with(SYMBOL, NOW)

    async def test_refresh_evaluates_bd_prefilter_once(self):
        await self.call_after(
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
        result = await self.call_after(
            make_bars(), prior_positions={SYMBOL: held}, positions={SYMBOL: held}
        )

        self.assertEqual(
            result.observations[0][1].strategy, (trading.StrategyTag.BD,)
        )
        self.data.bd_oi_windows.assert_awaited_once_with(SYMBOL, NOW)

    @patch.object(
        trading,
        "sample_probability",
        return_value={"chebyshev_upper_bound": 0.01},
    )
    async def test_same_cycle_long_close_then_bd_drops_bz_cooldown(
        self, _probability
    ):
        bars = make_bars()
        self.data.wait_ready = AsyncMock(return_value=True)
        self.data.fetch_bars = AsyncMock(return_value=bars)
        held = trading.Position(
            take_profit=90.0,
            stop_loss=50.0,
            position_side=trading.PositionSide.LONG,
            entry_price=80.0,
            name=SYMBOL,
            date=20260916,
            strategy=(trading.StrategyTag.BZ,),
            guard=trading.OIStop(open_interest=0),
        )
        observation = make_observation(
            side=trading.OrderSide.BUY,
            strategy=(trading.StrategyTag.BZ,),
            earliest_open_timestamp=None,
        )

        after_states = []
        real_after = self.strategy.after_instrument

        async def capture_after(symbol, current_bars, current_state, prior, context):
            after_states.append(current_state)
            return await real_after(symbol, current_bars, current_state, prior, context)

        with tempfile.TemporaryDirectory() as directory:
            state = trading.TradingState(
                "AUTOBN", f"{directory}/autobn-state.json"
            )
            state.apply(
                trading.StatePatch(
                    positions=((SYMBOL, held),),
                    observations=((SYMBOL, observation),),
                )
            )
            executor = Mock()
            executor.REQUIRES_ORDER_JOURNAL = False
            executor.prepare = AsyncMock(
                side_effect=lambda intent: trading.PreparedOrder(
                    intent=intent,
                    quantity=1.0,
                    price=bars.price,
                    entry_price=held.entry_price,
                )
            )
            result = trading.ExecutionResult(
                status=trading.ExecutionStatus.FILLED,
                quantity=1.0,
                price=bars.price,
                entry_price=held.entry_price,
                ratio=1.0,
                original_quantity=1.0,
                remaining_quantity=0,
            )
            executor.submit = AsyncMock(return_value=result)
            executor.complete_result = AsyncMock(return_value=result)
            notifications = Mock()
            notifications.send = AsyncMock()
            engine = trading.TradingEngine(
                self.strategy,
                executor,
                state,
                notifications,
                Mock(),
                Mock(timeout=1),
                Mock(),
            )

            with patch.object(
                self.strategy,
                "after_instrument",
                new=AsyncMock(side_effect=capture_after),
            ):
                outcome = await engine.process_instrument(SYMBOL, self.context)
            snapshot = state.snapshot()

        self.assertEqual(len(after_states), 1)
        state_before_after = after_states[0]
        self.assertNotIn(SYMBOL, state_before_after.positions)
        cooling_bz = state_before_after.observations[SYMBOL]
        self.assertEqual(cooling_bz.strategy, (trading.StrategyTag.BZ,))
        self.assertEqual(
            cooling_bz.earliest_open_timestamp,
            NOW.timestamp() + self.strategy.REOPEN_COOLDOWN_SECONDS,
        )
        self.assertEqual(outcome, trading.ScanOutcome.PROCESSED)
        self.assertNotIn(SYMBOL, snapshot.positions)
        replacement = snapshot.observations[SYMBOL]
        self.assertEqual(replacement.strategy, (trading.StrategyTag.BD,))
        self.assertIsNone(replacement.earliest_open_timestamp)
        self.data.bd_oi_windows.assert_awaited_once_with(SYMBOL, NOW)


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

    async def test_long_close_preserves_preexisting_bd_without_cooldown(self):
        self.seed(
            self.cooling_bz(earliest_open_timestamp=None),
            self.position(trading.PositionSide.LONG),
        )
        snapshot = await self.scan(make_bars())
        bd = snapshot.observations[SYMBOL]
        self.assertIn(SYMBOL, snapshot.positions)
        self.assertEqual(bd.strategy, (trading.StrategyTag.BD,))
        self.assertIsNone(bd.earliest_open_timestamp)
        self.executor.prepare.assert_not_awaited()

        self.data.oi_5m.return_value = [{"sumOpenInterest": 80.0}]
        self.data.bd_oi_windows.return_value = (
            OI_WINDOW[:-1] + (80.0,),
            OI_WINDOW[:-1],
        )
        snapshot = await self.scan(
            make_bars(close=99.0), NOW + datetime.timedelta(minutes=1)
        )
        self.assertNotIn(SYMBOL, snapshot.positions)
        self.assertEqual(snapshot.observations[SYMBOL], bd)
        self.assertEqual(
            self.executor.prepare.call_args.args[0].position.close_reason, "OI止损"
        )

        snapshot = await self.scan(
            make_bars(close=99.0), NOW + datetime.timedelta(minutes=5)
        )
        self.assertEqual(
            snapshot.positions[SYMBOL].position_side, trading.PositionSide.SHORT
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


if __name__ == "__main__":
    unittest.main()
