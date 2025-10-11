"""Kagent Claude SDK Integration.

This package provides integration between Kagent and Anthropic's Claude Agent SDK,
enabling Claude Code CLI-based agents to run in Kubernetes.
"""

from ._a2a import KAgentClaudeSDKApp
from ._agent_executor import ClaudeSDKAgentExecutor, ClaudeSDKAgentExecutorConfig

__all__ = [
    "KAgentClaudeSDKApp",
    "ClaudeSDKAgentExecutor",
    "ClaudeSDKAgentExecutorConfig",
]