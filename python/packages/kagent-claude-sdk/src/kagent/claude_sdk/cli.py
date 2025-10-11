"""CLI entrypoint for kagent-claude-sdk."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import uvicorn
from a2a.types import AgentCard, AgentCapabilities
from claude_agent_sdk import ClaudeAgentOptions
from pydantic import BaseModel, ValidationError

from ._a2a import KAgentClaudeSDKApp
from ._agent_executor import ClaudeSDKAgentExecutorConfig

logger = logging.getLogger(__name__)


class AgentConfigFile(BaseModel):
    """Configuration file format for Claude SDK agents."""

    system_prompt: str
    model: str | None = None
    mcp_servers: dict | str | Path | None = None
    max_turns: int | None = None
    continue_conversation: bool = True
    include_partial_messages: bool = True
    env: dict[str, str] | None = None


def load_config(config_path: Path) -> tuple[ClaudeAgentOptions, AgentCard]:
    """Load agent configuration from file."""
    with open(config_path) as f:
        config_data = json.load(f)

    try:
        agent_config = AgentConfigFile.model_validate(config_data)
    except ValidationError as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)

    options = ClaudeAgentOptions(
        system_prompt=agent_config.system_prompt,
        model=agent_config.model,
        mcp_servers=agent_config.mcp_servers,
        max_turns=agent_config.max_turns,
        continue_conversation=agent_config.continue_conversation,
        include_partial_messages=agent_config.include_partial_messages,
        env=agent_config.env or {},
    )

    # Create agent card from config
    agent_card = AgentCard(
        name=config_data.get("name", "claude-sdk-agent"),
        description=config_data.get("description", "A Claude SDK agent"),
        url="http://localhost:8080",  # Will be overridden by Kubernetes service
        version=config_data.get("version", "0.1.0"),
        capabilities=AgentCapabilities(
            streaming=agent_config.include_partial_messages,
            pushNotifications=False,
            stateTransitionHistory=True,
        ),
        skills=[],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )

    return options, agent_card


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Kagent Claude SDK Agent Runtime")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command - for dynamic agent loading
    run_parser = subparsers.add_parser("run", help="Run agent server")
    run_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    run_parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    run_parser.add_argument(
        "--config", default="/config/config.json", help="Config file path"
    )

    # Static command - for pre-configured agents
    static_parser = subparsers.add_parser("static", help="Run static agent server")
    static_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    static_parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    static_parser.add_argument(
        "--filepath", default="/config", help="Config directory path"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "run":
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)

        options, agent_card = load_config(config_path)

    elif args.command == "static":
        config_path = Path(args.filepath) / "config.json"
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)

        options, agent_card = load_config(config_path)

    # Get Kagent URL from environment
    kagent_url = os.getenv("KAGENT_URL", "http://kagent-controller.kagent:8083")
    app_name = os.getenv("KAGENT_NAME", "claude-sdk-agent")

    # Build FastAPI app
    kagent_app = KAgentClaudeSDKApp(
        options=options,
        agent_card=agent_card,
        kagent_url=kagent_url,
        app_name=app_name,
    )

    app = kagent_app.build()

    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
