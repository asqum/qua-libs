from __future__ import annotations

import unittest

from mcp import Client

from jy_agent.mcp_server import mcp


class McpSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_in_memory_client_lists_and_calls_read_only_tools(self) -> None:
        async with Client(mcp) as client:
            listing = await client.list_tools()
            names = {tool.name for tool in listing.tools}
            self.assertIn("jy_enter_measurement_mode", names)
            self.assertIn("jy_enter_autonomy_mode", names)
            self.assertIn("jy_continue_autonomy_mode", names)
            self.assertIn("jy_list_experiments", names)
            self.assertIn("jy_get_status", names)
            self.assertIn("jy_request_run", names)
            self.assertIn("jy_request_conversational_run", names)
            self.assertIn("jy_leave_measurement_mode", names)
            self.assertIn("jy_resume_measurement_mode", names)
            self.assertIn("jy_stop_workflow", names)
            self.assertIn("jy_apply_state_commit", names)
            self.assertIn("jy_request_drive_lo_recenter", names)
            self.assertIn("jy_request_initial_03a_zero_if", names)
            self.assertIn("jy_request_07b_prerequisites", names)
            self.assertIn("jy_request_07b_tail_power_reduction", names)
            self.assertIn("jy_reopen_07b_after_morphology_rule_change", names)
            self.assertIn("jy_request_03a_window_shift", names)
            self.assertIn("jy_request_03a_candidate_center", names)
            self.assertIn("jy_request_autonomy_lease", names)
            self.assertIn("jy_get_autonomy_status", names)
            self.assertIn("jy_wait_for_autonomy_status", names)
            self.assertIn("jy_autonomy_start_run", names)
            self.assertIn("jy_autonomy_analyze_run", names)
            self.assertIn("jy_autonomy_commit_state", names)
            self.assertIn("jy_autonomy_apply_setup_state", names)
            self.assertIn("jy_pause_autonomy", names)
            self.assertIn("jy_stop_autonomy", names)
            self.assertIn("jy_emergency_stop_autonomy", names)
            self.assertNotIn("jy_approve_proposal", names)
            result = await client.call_tool("jy_get_status", {})
            self.assertIn("server", result.structured_content)
            self.assertIn("measurement_mode", result.structured_content)
            self.assertIn("hardware_lock", result.structured_content)

    async def test_browser_approval_route_is_not_an_mcp_tool(self) -> None:
        app = mcp.streamable_http_app(streamable_http_path="/mcp")
        route_paths = {getattr(route, "path", None) for route in app.routes}
        self.assertIn("/healthz", route_paths)
        self.assertIn("/operator", route_paths)
        self.assertIn("/approve/{proposal_id}", route_paths)
        self.assertIn("/autonomy/{lease_id}", route_paths)
        listing = await mcp.list_tools()
        self.assertNotIn("jy_approve_proposal", {tool.name for tool in listing})


if __name__ == "__main__":
    unittest.main()
