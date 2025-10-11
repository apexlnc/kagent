"""Message converters between A2A and Claude SDK formats.

This module handles bidirectional conversion between:
- A2A (Agent-to-Agent) protocol messages
- Claude Agent SDK message types

Key responsibilities:
1. Convert user messages from A2A to Claude SDK format (simple text prompts)
2. Convert Claude SDK responses to A2A events for distribution

Message Type Mapping:
--------------------
Claude SDK -> A2A:
- UserMessage: Skipped (echoed messages)
- AssistantMessage: Maps to TaskStatusUpdateEvent with text/tool content
  - TextBlock -> TextPart (complete/final text)
  - ThinkingBlock -> TextPart with metadata (Claude's reasoning)
  - ToolUseBlock -> DataPart (function call requests)
  - ToolResultBlock -> DataPart (should rarely appear, logged as warning)
- StreamEvent: Maps to TaskStatusUpdateEvent with partial text deltas
- ResultMessage: Maps to TaskStatusUpdateEvent with session metadata (costs, duration, etc.)
- SystemMessage: Maps to TaskStatusUpdateEvent, errors become failed state

Streaming Behavior:
------------------
When include_partial_messages=True:
- StreamEvent: Contains partial text deltas as they arrive
- AssistantMessage: Contains complete final text (may duplicate streamed content)
- UI should handle deduplication using stable message IDs

When include_partial_messages=False:
- No StreamEvents
- AssistantMessage: Contains complete text only

Per Claude Agent SDK docs:
https://github.com/anthropics/claude-agent-sdk-python
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from a2a.types import (
    DataPart,
    Message,
    Part,
    Role,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from kagent.core.a2a._consts import (
    A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
    A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE,
    A2A_DATA_PART_METADATA_TYPE_KEY,
    get_kagent_metadata_key,
)

logger = logging.getLogger(__name__)


def convert_a2a_message_to_claude_sdk(a2a_message: Message) -> str:
    """Convert A2A message to Claude SDK prompt string."""
    text_parts = []
    for part in a2a_message.parts:
        if hasattr(part.root, "text"):
            text_parts.append(part.root.text)
    return " ".join(text_parts)


def convert_claude_sdk_message_to_a2a(
    claude_message: Any,
    task_id: str,
    context_id: str,
    app_name: str,
) -> list[TaskStatusUpdateEvent]:
    """Convert Claude SDK message to A2A events."""
    a2a_events = []

    # Handle different Claude SDK message types
    if isinstance(claude_message, UserMessage):
        # UserMessage contains tool results and user prompts
        # For tool results, we should process them
        a2a_parts = []

        for block in claude_message.content if isinstance(claude_message.content, list) else []:
            if isinstance(block, ToolResultBlock):
                # Tool execution results - these come from the Claude Code runtime
                a2a_parts.append(
                    Part(
                        DataPart(
                            data={
                                "tool_use_id": block.tool_use_id,
                                "content": block.content,
                                "is_error": block.is_error,
                            },
                            metadata={
                                get_kagent_metadata_key(
                                    A2A_DATA_PART_METADATA_TYPE_KEY
                                ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
                            },
                        )
                    )
                )
            elif isinstance(block, TextBlock):
                # User text content (echoed prompts or additional context)
                # Usually we skip these as they're echoed, but log for visibility
                logger.debug(f"UserMessage TextBlock: {block.text[:100]}...")

        if a2a_parts:
            # Only emit if we have tool results
            a2a_events.append(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    status=TaskStatus(
                        state=TaskState.working,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        message=Message(
                            message_id=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=a2a_parts,
                        ),
                    ),
                    context_id=context_id,
                    final=False,
                )
            )

    elif isinstance(claude_message, AssistantMessage):
        # Convert assistant response
        # When include_partial_messages=True, text is sent via StreamEvent deltas AND in final AssistantMessage
        # When include_partial_messages=False, text only appears in AssistantMessage
        # We should include ALL content types, letting the UI handle deduplication if needed
        a2a_parts = []

        for block in claude_message.content:
            if isinstance(block, TextBlock):
                # SKIP text blocks when streaming is enabled (default behavior)
                # Text was already sent via StreamEvent deltas
                # Only include TextBlock if we detect this is a non-streaming response
                # (but we don't have that context here, so skip for now)
                # TODO: Pass streaming flag to converter to handle both modes correctly
                pass

            elif isinstance(block, ThinkingBlock):
                # ThinkingBlock has both 'thinking' and 'signature' fields per SDK docs
                thinking_text = f"[Thinking: {block.thinking}]"
                if hasattr(block, 'signature') and block.signature:
                    thinking_text += f" (signature: {block.signature})"

                a2a_parts.append(
                    Part(
                        TextPart(
                            text=thinking_text,
                            metadata={get_kagent_metadata_key("thought"): True},
                        )
                    )
                )

            elif isinstance(block, ToolUseBlock):
                # ToolUseBlock appears in AssistantMessage when Claude requests tool execution
                a2a_parts.append(
                    Part(
                        DataPart(
                            data={
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            },
                            metadata={
                                get_kagent_metadata_key(
                                    A2A_DATA_PART_METADATA_TYPE_KEY
                                ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
                            },
                        )
                    )
                )

            elif isinstance(block, ToolResultBlock):
                # This should NEVER happen per SDK architecture
                # ToolResultBlock only appears in UserMessage (tool execution results)
                # If we hit this, there's a fundamental architectural problem
                raise ValueError(
                    f"ToolResultBlock found in AssistantMessage - architectural violation. "
                    f"Per Claude Agent SDK docs, ToolResultBlock only appears in UserMessage. "
                    f"This indicates a serious bug in the message flow. "
                    f"Tool use ID: {block.tool_use_id}"
                )

        if a2a_parts:
            a2a_events.append(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    status=TaskStatus(
                        state=TaskState.working,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        message=Message(
                            message_id=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=a2a_parts,
                        ),
                    ),
                    context_id=context_id,
                    final=False,
                    metadata={
                        get_kagent_metadata_key("app_name"): app_name,
                        get_kagent_metadata_key("model"): claude_message.model or "unknown",
                    },
                )
            )

    elif isinstance(claude_message, StreamEvent):
        # Handle streaming events for partial updates
        # StreamEvent has 'event' dict with 'type' key
        event_type = claude_message.event.get("type", "unknown") if claude_message.event else "unknown"
        logger.debug(f"Received stream event: {event_type}")

        # Check for content block delta events (partial text streaming)
        if event_type == "content_block_delta" and claude_message.event.get("delta", {}).get("type") == "text_delta":
            text = claude_message.event.get("delta", {}).get("text", "")
            if text:
                # Use content block index to create stable message ID for streaming updates
                content_index = claude_message.event.get("index", 0)
                # Create stable message_id using session_id + content_index so all deltas update the same message
                message_id = f"{claude_message.session_id}-content-{content_index}"

                a2a_events.append(
                    TaskStatusUpdateEvent(
                        task_id=task_id,
                        status=TaskStatus(
                            state=TaskState.working,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            message=Message(
                                message_id=message_id,
                                role=Role.agent,
                                parts=[Part(TextPart(text=text))],
                            ),
                        ),
                        context_id=context_id,
                        final=False,
                        metadata={
                            get_kagent_metadata_key("streaming"): True,
                            get_kagent_metadata_key("stream_delta"): True,  # Mark as delta for UI
                        },
                    )
                )

    elif isinstance(claude_message, ResultMessage):
        # ResultMessage indicates session completion with metrics
        # Per SDK docs: "ResultMessage: Session completion with cost/usage"
        # This contains final metadata but is NOT a final status update itself
        # The actual completion event is sent by the executor after processing all messages

        cost_str = f"${claude_message.total_cost_usd:.4f}" if claude_message.total_cost_usd is not None else "N/A"

        # Log the result field which may contain error details
        if claude_message.is_error:
            logger.error(
                f"Task FAILED. Cost: {cost_str}, "
                f"Duration: {claude_message.duration_ms}ms, "
                f"API Duration: {claude_message.duration_api_ms}ms, "
                f"Turns: {claude_message.num_turns}, "
                f"Session: {claude_message.session_id}, "
                f"Result: {claude_message.result}"
            )
        else:
            logger.info(
                f"Task completed. Cost: {cost_str}, "
                f"Duration: {claude_message.duration_ms}ms, "
                f"API Duration: {claude_message.duration_api_ms}ms, "
                f"Turns: {claude_message.num_turns}, "
                f"Session: {claude_message.session_id}, "
                f"Error: {claude_message.is_error}"
            )

        # Store session_id and all available metrics in metadata for future use
        metadata = {
            get_kagent_metadata_key("session_id"): claude_message.session_id,
            get_kagent_metadata_key("duration_ms"): claude_message.duration_ms,
            get_kagent_metadata_key("duration_api_ms"): claude_message.duration_api_ms,
            get_kagent_metadata_key("num_turns"): claude_message.num_turns,
            get_kagent_metadata_key("is_error"): claude_message.is_error,
        }
        if claude_message.total_cost_usd is not None:
            metadata[get_kagent_metadata_key("total_cost_usd")] = claude_message.total_cost_usd
        if claude_message.usage:
            metadata[get_kagent_metadata_key("usage")] = claude_message.usage
        if claude_message.result:
            metadata[get_kagent_metadata_key("result")] = claude_message.result

        # ResultMessage is metadata-only, not a final status change
        # The executor will determine final state based on is_error and accumulated results
        a2a_events.append(
            TaskStatusUpdateEvent(
                task_id=task_id,
                status=TaskStatus(
                    state=TaskState.working,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                context_id=context_id,
                final=False,  # Correct: this is just metadata, not the final status
                metadata=metadata,
            )
        )

    elif isinstance(claude_message, SystemMessage):
        # Handle system messages
        logger.info(f"System message: {claude_message.subtype} - {claude_message.data}")

        # ALWAYS log system messages for debugging
        if claude_message.subtype == "error":
            logger.error(f"Claude Code CLI error: {claude_message.data}")
            # Convert errors to failed state
            a2a_events.append(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        message=Message(
                            message_id=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=[Part(TextPart(text=f"Claude Code CLI Error: {claude_message.data}"))],
                        ),
                    ),
                    context_id=context_id,
                    final=True,
                )
            )
        else:
            # Log non-error system messages for visibility
            logger.info(f"System message ({claude_message.subtype}): {claude_message.data}")

    return a2a_events
