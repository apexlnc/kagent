"""FastAPI application builder for Claude SDK agents with A2A integration."""

import faulthandler
import logging
import os
from typing import Optional

import httpx
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard
from claude_agent_sdk import ClaudeAgentOptions
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from kagent.core import configure_tracing
from kagent.core.a2a import KAgentRequestContextBuilder, KAgentTaskStore

from ._agent_executor import ClaudeSDKAgentExecutor, ClaudeSDKAgentExecutorConfig
from ._token import KAgentTokenService

logger = logging.getLogger(__name__)


def health_check(request: Request) -> PlainTextResponse:
    """Health check endpoint."""
    return PlainTextResponse("OK")


def thread_dump(request: Request) -> PlainTextResponse:
    """Thread dump endpoint for debugging."""
    import io

    buf = io.StringIO()
    faulthandler.dump_traceback(file=buf)
    buf.seek(0)
    return PlainTextResponse(buf.read())


class KAgentClaudeSDKApp:
    """Kagent application builder for Claude SDK agents."""

    def __init__(
        self,
        options: ClaudeAgentOptions,
        agent_card: AgentCard,
        kagent_url: str,
        app_name: str,
        config: Optional[ClaudeSDKAgentExecutorConfig] = None,
        tracing: bool = True,
    ):
        self.options = options
        self.kagent_url = kagent_url
        self.app_name = app_name
        self.agent_card = agent_card
        self.config = config or ClaudeSDKAgentExecutorConfig()
        self._enable_tracing = tracing

    def build(self) -> FastAPI:
        """Build FastAPI application with A2A integration."""
        # Setup HTTP client with authentication
        token_service = KAgentTokenService(self.app_name)
        kagent_url_override = os.getenv("KAGENT_URL")
        http_client = httpx.AsyncClient(
            base_url=kagent_url_override or self.kagent_url,
            event_hooks=token_service.event_hooks(),
        )

        # Create agent executor
        agent_executor = ClaudeSDKAgentExecutor(
            options=self.options,
            app_name=self.app_name,
            config=self.config,
        )

        # Create task store
        kagent_task_store = KAgentTaskStore(http_client)

        # Create request context builder
        request_context_builder = KAgentRequestContextBuilder(task_store=kagent_task_store)

        # Create request handler
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=kagent_task_store,
            request_context_builder=request_context_builder,
        )

        # Create A2A FastAPI application
        a2a_app = A2AFastAPIApplication(
            agent_card=self.agent_card,
            http_handler=request_handler,
        )

        # Enable fault handler for debugging
        faulthandler.enable()

        # Create main FastAPI app
        app = FastAPI(lifespan=token_service.lifespan())

        # Configure tracing/instrumentation if enabled
        if self._enable_tracing:
            try:
                configure_tracing(app)
                logger.info("Tracing configured for KAgent Claude SDK app")
            except Exception:
                logger.exception("Failed to configure tracing")

        # Add health check and debugging routes
        app.add_route("/health", methods=["GET"], route=health_check)
        app.add_route("/thread_dump", methods=["GET"], route=thread_dump)

        # Add A2A routes
        a2a_app.add_routes_to_app(app)

        return app
