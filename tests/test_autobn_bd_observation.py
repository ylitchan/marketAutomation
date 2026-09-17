import datetime
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

    async def test_bz_wins_without_requesting_bd_oi_and_keeps_cooldown(self):
        existing = make_observation()
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


if __name__ == "__main__":
    unittest.main()
