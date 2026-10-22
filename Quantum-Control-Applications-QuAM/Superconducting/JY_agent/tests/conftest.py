"""Test-wide settings for the JY_agent suite."""

from jy_agent.service import AgentService

# The static history site is written synchronously in tests, so no background
# thread writes into a temporary folder that is being deleted (2026-10-22).
AgentService.HISTORY_SITE_ASYNC = False
