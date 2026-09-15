import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from perk_pushplus import Channel, Template

import autoTrade_pm_refactored as trading
from pushplus_notifications import (
    TradeNotification,
    _send_pushplus_sync,
    send_pushplus,
)


class PushPlusRequestTests(unittest.TestCase):
    @patch("pushplus_notifications._get_pushplus_client")
    def test_service_account_keeps_wechat_markdown_delivery(self, get_client):
        notification = TradeNotification("开仓成功", "# AUTOBN 开仓成功")

        _send_pushplus_sync(notification, "token", "secret", "pushplus")

        request = get_client.return_value.send.call_args.args[0]
        self.assertEqual(request.channel, Channel.WECHAT)
        self.assertIsNone(request.option)
        self.assertEqual(request.template, Template.MARKDOWN)
        self.assertEqual(request.title, notification.title)
        self.assertEqual(request.content, notification.content)

    @patch("pushplus_notifications._get_pushplus_client")
    def test_feishu_signal_uses_configured_pushplus_webhook(self, get_client):
        with patch.dict(os.environ, {"PUSHPLUS_FEISHU_OPTION": ""}):
            _send_pushplus_sync("AUTOBN signal", "token", "secret", "feishu")

        request = get_client.return_value.send.call_args.args[0]
        self.assertEqual(request.channel, Channel.WEBHOOK)
        self.assertEqual(request.option, "feishu")
        self.assertEqual(request.template, Template.TXT)
        self.assertIsNone(request.title)
        self.assertEqual(request.content, "AUTOBN signal")

    @patch("pushplus_notifications._get_pushplus_client")
    def test_wecom_signal_allows_option_override(self, get_client):
        with patch.dict(os.environ, {"PUSHPLUS_WECOM_OPTION": "wecom-production"}):
            _send_pushplus_sync("AUTOA signal", "token", "secret", "wecom")

        request = get_client.return_value.send.call_args.args[0]
        self.assertEqual(request.channel, Channel.WEBHOOK)
        self.assertEqual(request.option, "wecom-production")
        self.assertEqual(request.template, Template.TXT)

    def test_unknown_destination_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知 PushPlus 发送目标"):
            _send_pushplus_sync("message", "token", "secret", "unknown")


class PushPlusAsyncTests(unittest.IsolatedAsyncioTestCase):
    @patch("pushplus_notifications._send_pushplus_sync")
    async def test_async_sender_forwards_destination(self, send_sync):
        sent = await send_pushplus(
            "signal",
            token="token",
            secret_key="secret",
            channel="feishu",
        )

        self.assertTrue(sent)
        send_sync.assert_called_once_with("signal", "token", "secret", "feishu")


class NotificationsTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_destinations_delegate_to_pushplus(self):
        http = Mock(timeout=2)
        logger = Mock()
        notifications = trading.Notifications(http, logger)
        cases = (
            (TradeNotification("开仓成功", "content"), "pushplus"),
            ("AUTOBN signal", "feishu"),
            ("AUTOA signal", "wecom"),
        )

        for message, channel in cases:
            with (
                self.subTest(channel=channel),
                patch.object(
                    trading,
                    "send_pushplus",
                    new=AsyncMock(return_value=True),
                ) as sender,
            ):
                await notifications.send(message, channel=channel)
                sender.assert_awaited_once_with(message, channel=channel)


if __name__ == "__main__":
    unittest.main()
