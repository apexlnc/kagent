"""Tests for message converters."""

import pytest
from a2a.types import Message, Part, TextPart, TaskState
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from kagent.claude_sdk._converters import (
    convert_a2a_message_to_claude_sdk,
    convert_claude_sdk_message_to_a2a,
)


class TestMessageConverters:
    """Test cases for Claude SDK message converters.

    Minimal but comprehensive coverage:
    1. Core conversion functionality
    2. Critical architectural invariants
    3. Complex scenarios (multiple content types)
    4. Key SDK features (streaming, metadata)
    """

    def test_convert_a2a_message_to_claude_sdk(self):
        """Test converting A2A message to Claude SDK prompt string."""
        message = Message(
            message_id="test-id", role="user", parts=[Part(TextPart(text="Hello, world!"))]
        )

        result = convert_a2a_message_to_claude_sdk(message)

        assert result == "Hello, world!"


    def test_convert_assistant_message_with_multiple_content_types(self):
        """Test AssistantMessage with all content types (TextBlock, ThinkingBlock, ToolUseBlock).

        This single test covers:
        - TextBlock capture (was being dropped - critical fix)
        - ThinkingBlock with signature field (was incomplete)
        - ToolUseBlock for tool requests
        """
        claude_message = AssistantMessage(
            content=[
                TextBlock(text="I'll help with that."),
                ThinkingBlock(thinking="Need to calculate this", signature="sig-123"),
                ToolUseBlock(id="tool-1", name="calculator", input={"op": "add"}),
            ],
            model="claude-3-5-sonnet",
        )

        events = convert_claude_sdk_message_to_a2a(
            claude_message, task_id="task-1", context_id="ctx-1", app_name="test"
        )

        assert len(events) == 1
        parts = events[0].status.message.parts
        assert len(parts) == 3

        # TextBlock - must be captured
        assert parts[0].root.text == "I'll help with that."

        # ThinkingBlock - must include signature
        assert "[Thinking:" in parts[1].root.text
        assert "sig-123" in parts[1].root.text
        assert parts[1].metadata.get("kagent_thought") is True

        # ToolUseBlock - function call
        assert parts[2].root.data["name"] == "calculator"
        assert parts[2].metadata.get("kagent_type") == "function_call"

    def test_convert_user_message_with_tool_result(self):
        """Test UserMessage with ToolResultBlock (correct architecture).

        ToolResultBlock ONLY appears in UserMessage per SDK docs.
        """
        claude_message = UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id="tool-1",
                    content="Result: 42",
                    is_error=False,
                )
            ],
        )

        events = convert_claude_sdk_message_to_a2a(
            claude_message, task_id="task-1", context_id="ctx-1", app_name="test"
        )

        assert len(events) == 1
        part = events[0].status.message.parts[0]
        assert part.root.data["tool_use_id"] == "tool-1"
        assert part.root.data["content"] == "Result: 42"
        assert part.root.data["is_error"] is False
        assert part.metadata.get("kagent_type") == "function_response"

    def test_tool_result_block_in_assistant_message_raises_error(self):
        """CRITICAL: Enforce architectural invariant.

        ToolResultBlock in AssistantMessage is an architectural violation.
        Fail fast to catch bugs immediately.
        """
        claude_message = AssistantMessage(
            content=[
                ToolResultBlock(
                    tool_use_id="tool-1",
                    content="Result: 3",
                    is_error=False,
                )
            ],
            model="claude-3-5-sonnet",
        )

        with pytest.raises(ValueError, match="architectural violation"):
            convert_claude_sdk_message_to_a2a(
                claude_message, task_id="task-1", context_id="ctx-1", app_name="test"
            )

    def test_convert_stream_event(self):
        """Test StreamEvent with correct SDK structure.

        Critical fixes:
        - Uses 'event' dict (not 'data')
        - Has 'uuid' and 'session_id' fields
        - Stable message IDs prevent UI jitter
        """
        claude_message = StreamEvent(
            uuid="evt-123",
            session_id="session-1",
            event={
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": "Streaming text..."
                }
            },
            parent_tool_use_id=None,
        )

        events = convert_claude_sdk_message_to_a2a(
            claude_message, task_id="task-1", context_id="ctx-1", app_name="test"
        )

        assert len(events) == 1
        assert events[0].status.message.parts[0].root.text == "Streaming text..."
        assert events[0].metadata.get("kagent_streaming") is True
        assert events[0].metadata.get("kagent_stream_delta") is True

    def test_convert_result_message(self):
        """Test ResultMessage captures complete session metadata.

        Critical fixes:
        - total_cost_usd (was using wrong field name)
        - ALL 9+ fields captured (was only capturing 3)
        - Includes: duration_api_ms, num_turns, is_error, usage, result
        """
        claude_message = ResultMessage(
            subtype="complete",
            duration_ms=500,
            duration_api_ms=450,
            is_error=False,
            num_turns=3,
            session_id="session-1",
            total_cost_usd=0.001,
            usage={"input_tokens": 100, "output_tokens": 50},
            result="Success",
        )

        events = convert_claude_sdk_message_to_a2a(
            claude_message, task_id="task-1", context_id="ctx-1", app_name="test"
        )

        # Validate all critical fields are captured
        assert len(events) == 1
        metadata = events[0].metadata
        assert metadata.get("kagent_session_id") == "session-1"
        assert metadata.get("kagent_total_cost_usd") == 0.001
        assert metadata.get("kagent_duration_ms") == 500
        assert metadata.get("kagent_duration_api_ms") == 450
        assert metadata.get("kagent_num_turns") == 3
        assert metadata.get("kagent_is_error") is False
        assert metadata.get("kagent_usage") == {"input_tokens": 100, "output_tokens": 50}
        assert metadata.get("kagent_result") == "Success"
        assert events[0].final is False  # Metadata only, not final status
