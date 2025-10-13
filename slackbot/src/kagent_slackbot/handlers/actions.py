"""Action (button) handlers"""

from typing import Any
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient
from structlog import get_logger

from ..services.a2a_client import A2AClient

logger = get_logger(__name__)


def register_action_handlers(app: AsyncApp, a2a_client: A2AClient) -> None:
    """Register action handlers for interactive buttons"""

    @app.action("approval_approve")
    async def handle_approval_approve(
        ack: Any,
        action: dict[str, Any],
        body: dict[str, Any],
        client: AsyncWebClient,
    ) -> None:
        """Handle approval button click"""
        await ack()

        button_value = action["value"]
        parts = button_value.split("|")
        session_id = parts[0]
        agent_full_name = parts[1] if len(parts) > 1 else ""

        user_id = body["user"]["id"]
        channel = body["container"]["channel_id"]
        message_ts = body["container"]["message_ts"]

        logger.info(
            "User approved action",
            user=user_id,
            session=session_id,
            agent=agent_full_name,
        )

        # Send approval message back to agent in same session
        if "/" in agent_full_name:
            namespace, agent_name = agent_full_name.split("/", 1)

            try:
                await a2a_client.invoke_agent(
                    namespace=namespace,
                    agent_name=agent_name,
                    message="User approved: proceed with the action.",
                    session_id=session_id,
                    user_id=user_id,
                )

                await client.chat_update(
                    channel=channel,
                    ts=message_ts,
                    text="✅ Approved - Agent will proceed",
                    blocks=body["message"]["blocks"]
                    + [
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": "✅ _User approved - agent proceeding_",
                                }
                            ],
                        }
                    ],
                )

                logger.info("Approval sent to agent", session=session_id, agent=agent_full_name)

            except Exception as e:
                logger.error("Failed to send approval", error=str(e), session=session_id)
                await client.chat_postEphemeral(
                    channel=channel,
                    user=user_id,
                    text=f"❌ Failed to send approval to agent: {str(e)}",
                )

    @app.action("approval_deny")
    async def handle_approval_deny(
        ack: Any,
        action: dict[str, Any],
        body: dict[str, Any],
        client: AsyncWebClient,
    ) -> None:
        """Handle denial button click"""
        await ack()

        button_value = action["value"]
        parts = button_value.split("|")
        session_id = parts[0]
        agent_full_name = parts[1] if len(parts) > 1 else ""

        user_id = body["user"]["id"]
        channel = body["container"]["channel_id"]
        message_ts = body["container"]["message_ts"]

        logger.info(
            "User denied action",
            user=user_id,
            session=session_id,
            agent=agent_full_name,
        )

        # Send denial message back to agent
        if "/" in agent_full_name:
            namespace, agent_name = agent_full_name.split("/", 1)

            try:
                await a2a_client.invoke_agent(
                    namespace=namespace,
                    agent_name=agent_name,
                    message="User denied: cancel the action and do not proceed.",
                    session_id=session_id,
                    user_id=user_id,
                )

                await client.chat_update(
                    channel=channel,
                    ts=message_ts,
                    text="❌ Denied - Agent will not proceed",
                    blocks=body["message"]["blocks"]
                    + [
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": "❌ _User denied - agent canceled_",
                                }
                            ],
                        }
                    ],
                )

                logger.info("Denial sent to agent", session=session_id, agent=agent_full_name)

            except Exception as e:
                logger.error("Failed to send denial", error=str(e), session=session_id)
                await client.chat_postEphemeral(
                    channel=channel,
                    user=user_id,
                    text=f"❌ Failed to send denial to agent: {str(e)}",
                )
