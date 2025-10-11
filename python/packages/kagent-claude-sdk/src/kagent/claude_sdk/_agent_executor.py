"""Claude SDK Agent Executor for A2A protocol integration."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from pydantic import BaseModel

from kagent.core.a2a import TaskResultAggregator
from kagent.core.a2a._consts import get_kagent_metadata_key

from ._converters import convert_a2a_message_to_claude_sdk, convert_claude_sdk_message_to_a2a

logger = logging.getLogger(__name__)


class ClaudeSDKAgentExecutorConfig(BaseModel):
    """Configuration for the ClaudeSDKAgentExecutor."""

    execution_timeout: float = 300.0  # 5 minutes default
    include_partial_messages: bool = True
    continue_conversation: bool = True
    max_turns: Optional[int] = None


class ClaudeSDKAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs Claude Agent SDK against A2A requests."""

    def __init__(
        self,
        *,
        options: ClaudeAgentOptions,
        app_name: str,
        config: Optional[ClaudeSDKAgentExecutorConfig] = None,
    ):
        super().__init__()
        self._options = options
        self.app_name = app_name
        self._config = config or ClaudeSDKAgentExecutorConfig()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        """Execute the agent using Claude SDK."""
        if not context.message:
            raise ValueError("A2A request must have a message")

        # Emit submitted event for new tasks
        if not context.current_task:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    status=TaskStatus(
                        state=TaskState.submitted,
                        message=context.message,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    context_id=context.context_id,
                    final=False,
                )
            )

        # Emit working event
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                status=TaskStatus(
                    state=TaskState.working,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                context_id=context.context_id,
                final=False,
                metadata={
                    get_kagent_metadata_key("app_name"): self.app_name,
                    get_kagent_metadata_key("session_id"): context.context_id,
                },
            )
        )

        try:
            await self._handle_request(context, event_queue)
        except Exception as e:
            logger.error(f"Error during Claude SDK execution: {e}", exc_info=True)
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        message=Message(
                            message_id=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=[Part(TextPart(text=str(e)))],
                        ),
                    ),
                    context_id=context.context_id,
                    final=True,
                )
            )

    async def _handle_request(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        """Handle the request by running Claude SDK client."""
        # Create a copy of options to avoid modifying the original
        options = self._options.model_copy(deep=True) if hasattr(self._options, 'model_copy') else self._options
        options.continue_conversation = self._config.continue_conversation
        options.include_partial_messages = self._config.include_partial_messages
        if self._config.max_turns:
            options.max_turns = self._config.max_turns

        # Determine session management strategy
        if context.current_task:
            # Continue existing session using context_id
            options.resume = context.context_id
        elif self._config.continue_conversation:
            # Continue most recent conversation
            options.continue_conversation = True
        else:
            # Start fresh session
            options.continue_conversation = False

        task_result_aggregator = TaskResultAggregator()
        session_id = None
        result_message_received = None  # Track the ResultMessage to check is_error

        # Create Claude SDK client with timeout
        # The async context manager automatically handles connection setup and teardown
        async with ClaudeSDKClient(options) as client:
            # Convert and send user message
            user_prompt = convert_a2a_message_to_claude_sdk(context.message)
            await client.query(user_prompt)

            # Stream responses with timeout
            try:
                # Apply timeout to entire streaming operation
                async with asyncio.timeout(self._config.execution_timeout):
                    async for message in client.receive_response():
                        # Track session ID and ResultMessage
                        if hasattr(message, 'session_id'):
                            session_id = message.session_id

                        # Track ResultMessage to check is_error flag
                        from claude_agent_sdk.types import ResultMessage
                        if isinstance(message, ResultMessage):
                            result_message_received = message

                        # Log the raw message for debugging
                        logger.debug(f"Received message type: {type(message).__name__}")

                        # Convert Claude SDK message to A2A events
                        a2a_events = convert_claude_sdk_message_to_a2a(
                            message, context.task_id, context.context_id, self.app_name
                        )

                        for a2a_event in a2a_events:
                            task_result_aggregator.process_event(a2a_event)
                            await event_queue.enqueue_event(a2a_event)

            except asyncio.TimeoutError:
                logger.error(
                    f"Execution timed out after {self._config.execution_timeout} seconds"
                )
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        task_id=context.task_id,
                        status=TaskStatus(
                            state=TaskState.failed,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            message=Message(
                                message_id=str(uuid.uuid4()),
                                role=Role.agent,
                                parts=[
                                    Part(
                                        TextPart(
                                            text=f"Execution timed out after {self._config.execution_timeout} seconds"
                                        )
                                    )
                                ],
                            ),
                        ),
                        context_id=context.context_id,
                        final=True,
                    )
                )
                return

        # Emit final result with session metadata
        final_metadata = {
            get_kagent_metadata_key("app_name"): self.app_name,
        }
        if session_id:
            final_metadata[get_kagent_metadata_key("session_id")] = session_id

        # Debug logging for final state
        logger.info(
            f"Final state: task_state={task_result_aggregator.task_state}, "
            f"has_message={task_result_aggregator.task_status_message is not None}, "
            f"has_parts={task_result_aggregator.task_status_message.parts if task_result_aggregator.task_status_message else 'N/A'}, "
            f"result_is_error={result_message_received.is_error if result_message_received else 'N/A'}"
        )

        # Determine success based on ResultMessage.is_error flag (most reliable)
        # With streaming, parts are sent via StreamEvents, so aggregator may not have parts
        is_success = (
            result_message_received is not None
            and not result_message_received.is_error
            and task_result_aggregator.task_state == TaskState.working
        )

        if is_success:
            # Success case - send completion without artifact if no parts accumulated
            # (streaming already sent the content)
            if (
                task_result_aggregator.task_status_message is not None
                and task_result_aggregator.task_status_message.parts
            ):
                # Send artifact if we have accumulated parts
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        task_id=context.task_id,
                        last_chunk=True,
                        context_id=context.context_id,
                        artifact=Artifact(
                            artifact_id=str(uuid.uuid4()),
                            parts=task_result_aggregator.task_status_message.parts,
                        ),
                    )
                )

            # Send completion status for success
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    status=TaskStatus(
                        state=TaskState.completed,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    context_id=context.context_id,
                    final=True,
                    metadata=final_metadata,
                )
            )
        else:
            # Failure case - ensure we always have a valid message
            failure_message = task_result_aggregator.task_status_message
            if failure_message is None or not failure_message.parts:
                # Create a default error message if none exists
                failure_message = Message(
                    message_id=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[Part(TextPart(text="Task failed without error details. Check agent logs."))],
                )

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    status=TaskStatus(
                        state=task_result_aggregator.task_state,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        message=failure_message,
                    ),
                    context_id=context.context_id,
                    final=True,
                    metadata=final_metadata,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancel the running Claude SDK execution."""
        # Emit cancellation event
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                status=TaskStatus(
                    state=TaskState.cancelled,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    message=Message(
                        message_id=str(uuid.uuid4()),
                        role=Role.agent,
                        parts=[Part(TextPart(text="Task cancelled by user"))],
                    ),
                ),
                context_id=context.context_id,
                final=True,
            )
        )
        # Note: Actual cancellation would require tracking the running client
        # and calling client.interrupt() - implement if needed