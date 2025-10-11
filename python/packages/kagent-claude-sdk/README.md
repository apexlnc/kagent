# Kagent Claude SDK Integration

Kagent integration for Anthropic's Claude Agent SDK, enabling Claude Code CLI-based agents in Kubernetes.

## Installation

```bash
uv pip install kagent-claude-sdk
```

## Requirements

- Python 3.11+
- Node.js 22+
- `@anthropic-ai/claude-code` CLI installed globally
- Kubernetes cluster with Kagent installed

## Quick Start

### Create a Claude SDK Agent

```python
from claude_agent_sdk import ClaudeAgentOptions
from kagent.claude_sdk import KAgentClaudeSDKApp

# Configure agent options
options = ClaudeAgentOptions(
    system_prompt="You are a helpful Kubernetes assistant.",
    model="claude-3-5-sonnet-20241022",
    mcp_servers={
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        }
    },
    max_turns=5,
)

# Create agent card
agent_card = {
    "name": "k8s-helper",
    "description": "Kubernetes helper agent using Claude SDK",
    "version": "0.1.0",
    "capabilities": {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": True,
    },
    "skills": [],
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
}

# Build FastAPI app
app_builder = KAgentClaudeSDKApp(
    options=options,
    agent_card=agent_card,
    kagent_url="http://kagent-controller.kagent:8083",
    app_name="k8s-helper",
)

app = app_builder.build()
```

### Run the Agent

```bash
kagent-claude-sdk run --config config.json --host 0.0.0.0 --port 8080
```

## Kubernetes Deployment

### Define Agent CRD

```yaml
apiVersion: kagent.dev/v1alpha2
kind: Agent
metadata:
  name: my-claude-agent
  namespace: kagent
spec:
  type: BYO
  description: My Claude SDK agent
  byo:
    deployment:
      image: cr.kagent.dev/kagent-dev/kagent/claude-sdk-app:latest
      env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: anthropic-key
              key: api-key
```

### Apply to Cluster

```bash
kubectl apply -f agent.yaml
```

## Features

- **Claude Code CLI Integration**: Leverages Claude's official CLI for robust execution
- **MCP Tool Support**: Connect to any MCP server (stdio, SSE, HTTP)
- **Session Continuity**: Maintains conversation history across invocations
- **Streaming Responses**: Real-time streaming via A2A protocol
- **Kubernetes Native**: First-class support with BYO agent type

## Configuration

### AgentOptions

- `system_prompt`: System prompt for the agent
- `model`: Model to use (e.g., "claude-3-5-sonnet-20241022")
- `mcp_servers`: Dictionary of MCP server configurations
- `max_turns`: Maximum agentic turns per invocation
- `continue_conversation`: Enable session continuity
- `include_partial_messages`: Enable streaming
- `env`: Environment variables passed to Claude Code CLI subprocess

### Deployment Options

- `replicas`: Number of pod replicas
- `image`: Full image reference
- `resources`: CPU/memory resource requirements
- `env`: Environment variables (including ANTHROPIC_API_KEY)

### Proxy Configuration

To route Anthropic API calls through a proxy, set the `ANTHROPIC_BASE_URL` environment variable:

**Option 1: Via Kubernetes Deployment**
```yaml
apiVersion: kagent.dev/v1alpha2
kind: Agent
metadata:
  name: my-claude-agent
spec:
  type: BYO
  byo:
    deployment:
      image: cr.kagent.dev/kagent-dev/kagent/claude-sdk-app:latest
      env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: anthropic-key
              key: api-key
        - name: ANTHROPIC_BASE_URL
          value: "https://your-proxy.company.com"
```

**Option 2: Via Config File**
```json
{
  "name": "my-agent",
  "system_prompt": "You are a helpful assistant.",
  "model": "claude-3-5-sonnet-20241022",
  "env": {
    "ANTHROPIC_BASE_URL": "https://your-proxy.company.com"
  }
}
```

**Option 3: Programmatically**
```python
options = ClaudeAgentOptions(
    system_prompt="You are a helpful assistant.",
    model="claude-3-5-sonnet-20241022",
    env={
        "ANTHROPIC_BASE_URL": "https://your-proxy.company.com",
    }
)
```

The Claude Code CLI subprocess will inherit these environment variables and use the proxy URL for all Anthropic API calls.

## Examples

See `python/samples/claude-sdk/` for complete examples:
- Basic agent
- Agent with MCP tools
- Multi-turn conversation agent

## Architecture

Claude SDK agents run as Kubernetes pods with:
- Python runtime for Kagent integration
- Node.js runtime for Claude Code CLI
- FastAPI server for A2A protocol
- AgentGateway for MCP proxying (optional)

## Troubleshooting

### Agent Pod Not Starting

Check pod logs:
```bash
kubectl logs -n kagent deployment/my-claude-agent
```

Common issues:
- Missing Node.js or Claude CLI
- Invalid ANTHROPIC_API_KEY
- MCP server connection failures

### Session Not Persisting

Verify:
- `continue_conversation: true` in config
- Kagent database is running
- Session ID is consistent across requests

## Contributing

See [CONTRIBUTION.md](../../../CONTRIBUTION.md) for guidelines.

## License

Apache 2.0