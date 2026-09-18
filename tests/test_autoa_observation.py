import datetime
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock

import autoTrade_pm_refactored as trading

SYMBOL = "600000"
NOW = datetime.datetime(2026, 9, 18, 12, 0, tzinfo=datetime.UTC)


def make_observation(*, strategy=(trading.StrategyTag.BZ, trading.StrategyTag.N)):
    return trading.Observation(
        price=90.0,
        timestamp=NOW.timestamp() - 86400,
        side=trading.OrderSide.BUY,
        strategy=strategy,
        name="测试标的",
        bz_reference_high=123.4,
        earliest_open_timestamp=NOW.timestamp() + 3600,
    )


def make_position(*, strategy=(trading.StrategyTag.BZ, trading.StrategyTag.N)):
    return trading.Position(
        take_profit=110.0,
        stop_loss=80.0,
        position_side=trading.PositionSide.LONG,
        entry_price=100.0,
        name="测试标的",
        date=20260917,
        strategy=strategy,
    )


class AUTOAObservationCloseTests(unittest.TestCase):
    def setUp(self):
        self.strategy = trading.AUTOA(Mock())
        self.position = make_position()
        self.intent = trading.TradeIntent(
            trading.ActionKind.CLOSE,
            SYMBOL,
            self.position,
            105.0,
        )

    def test_full_close_removes_n_and_preserves_observation_state(self):
        observation = make_observation(
            strategy=(
                trading.StrategyTag.BZ,
                trading.StrategyTag.N,
                trading.StrategyTag.DCA,
            )
        )
        state = trading.StateSnapshot({}, {SYMBOL: observation})

        result = self.strategy.observation_after_close(self.intent, state, NOW)

        self.assertEqual(
            result.strategy,
            (trading.StrategyTag.BZ, trading.StrategyTag.DCA),
        )
        self.assertEqual(result.price, observation.price)
        self.assertEqual(result.timestamp, observation.timestamp)
        self.assertEqual(result.side, observation.side)
        self.assertEqual(result.name, observation.name)
        self.assertEqual(result.bz_reference_high, observation.bz_reference_high)
        self.assertIsNone(result.earliest_open_timestamp)
        self.assertEqual(result.reopen_pending_date, "2026-09-18")

    def test_full_close_without_n_is_unchanged_except_cooldown(self):
        observation = make_observation(strategy=(trading.StrategyTag.BZ,))
        state = trading.StateSnapshot({}, {SYMBOL: observation})

        result = self.strategy.observation_after_close(self.intent, state, NOW)

        self.assertEqual(result.strategy, observation.strategy)
        self.assertIsNone(result.earliest_open_timestamp)
        self.assertEqual(result.reopen_pending_date, "2026-09-18")

    def test_missing_observation_keeps_empty_fallback_strategy(self):
        state = trading.StateSnapshot({}, {})

        result = self.strategy.observation_after_close(self.intent, state, NOW)

        self.assertEqual(result.strategy, ())
        self.assertEqual(result.price, self.intent.price)
        self.assertEqual(result.name, self.position.name)
        self.assertEqual(result.reopen_pending_date, "2026-09-18")


class AUTOAObservationCloseEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = trading.TradingState("AUTOA", f"{self.directory.name}/state.json")
        self.strategy = trading.AUTOA(Mock())
        self.executor = Mock()
        self.executor.REQUIRES_ORDER_JOURNAL = False
        self.executor.complete_result = AsyncMock(
            side_effect=lambda order, result: result
        )
        self.engine = trading.TradingEngine(
            self.strategy,
            self.executor,
            self.state,
            Mock(),
            Mock(),
            Mock(),
            Mock(),
        )
        self.position = make_position()
        self.observation = make_observation()
        self.intent = trading.TradeIntent(
            trading.ActionKind.CLOSE,
            SYMBOL,
            self.position,
            105.0,
        )
        self.order = trading.PreparedOrder(
            intent=self.intent,
            quantity=100,
            price=105.0,
            entry_price=100.0,
        )
        self.context = trading.MarketContext(now=NOW)

    def seed(self):
        self.state.apply(
            trading.StatePatch(
                positions=((SYMBOL, self.position),),
                observations=((SYMBOL, self.observation),),
            )
        )

    async def finish(self, result):
        self.seed()
        await self.engine._finish_execution(
            self.order, result, self.context, notify=False
        )
        return self.state.snapshot()

    async def test_confirmed_full_close_removes_n_from_observation(self):
        snapshot = await self.finish(
            trading.ExecutionResult(
                status=trading.ExecutionStatus.FILLED,
                quantity=100,
                price=105.0,
                entry_price=100.0,
                original_quantity=100,
                remaining_quantity=0,
            )
        )

        self.assertNotIn(SYMBOL, snapshot.positions)
        self.assertEqual(
            snapshot.observations[SYMBOL].strategy,
            (trading.StrategyTag.BZ,),
        )

    async def test_failed_close_keeps_n(self):
        snapshot = await self.finish(
            trading.ExecutionResult(status=trading.ExecutionStatus.FAILED)
        )

        self.assertIn(SYMBOL, snapshot.positions)
        self.assertEqual(
            snapshot.observations[SYMBOL].strategy,
            (trading.StrategyTag.BZ, trading.StrategyTag.N),
        )

    async def test_unknown_close_keeps_n(self):
        snapshot = await self.finish(
            trading.ExecutionResult(status=trading.ExecutionStatus.UNKNOWN)
        )

        self.assertIn(SYMBOL, snapshot.positions)
        self.assertEqual(
            snapshot.observations[SYMBOL].strategy,
            (trading.StrategyTag.BZ, trading.StrategyTag.N),
        )

    async def test_partial_close_keeps_n(self):
        snapshot = await self.finish(
            trading.ExecutionResult(
                status=trading.ExecutionStatus.FILLED,
                quantity=50,
                price=105.0,
                entry_price=100.0,
                ratio=0.5,
                original_quantity=100,
                remaining_quantity=50,
            )
        )

        self.assertIn(SYMBOL, snapshot.positions)
        self.assertEqual(
            snapshot.observations[SYMBOL].strategy,
            (trading.StrategyTag.BZ, trading.StrategyTag.N),
        )


if __name__ == "__main__":
    unittest.main()
