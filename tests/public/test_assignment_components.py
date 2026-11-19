from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from pathlib import Path

from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.rate_limiter import RateLimitPlugin


def test_rate_limiter_sliding_window_is_per_user(monkeypatch):
    clock = iter((100.0, 100.5, 100.75, 101.0, 101.0))
    plugin = RateLimitPlugin(max_requests=2, window_seconds=1)
    monkeypatch.setattr(plugin, "_clock", lambda: next(clock))

    async def call(user_id: str):
        return await plugin.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=user_id),
            user_message=None,
        )

    assert asyncio.run(call("user-a")) is None
    assert asyncio.run(call("user-a")) is None
    assert asyncio.run(call("user-a")) is not None
    assert asyncio.run(call("user-a")) is None
    assert asyncio.run(call("user-b")) is None
    assert plugin.blocked_count == 1


def test_audit_log_correlates_input_and_output(tmp_path):
    audit = AuditLogPlugin()
    audit.record_input(user_id="customer-1", text="Check my balance")
    audit.record_output(
        user_id="customer-1",
        text="Request blocked",
        blocked=True,
        layer="input_guardrail",
    )

    path = audit.export_json(tmp_path / "audit.json")
    saved = path.read_text(encoding="utf-8")
    assert '"input": "Check my balance"' in saved
    assert '"layer": "input_guardrail"' in saved
    assert '"blocked": true' in saved
    assert audit.logs[0]["latency_ms"] >= 0


def test_monitoring_recomputes_threshold_alerts():
    monitor = MonitoringAlert(block_rate_threshold=0.25)
    monitor.total_requests = 4
    monitor.blocked_requests = 2
    assert [alert.metric for alert in monitor.check_metrics()] == ["block_rate"]

    monitor.blocked_requests = 0
    assert monitor.check_metrics() == []


def test_production_plugin_order():
    from assignment.pipeline import build_production_plugins

    plugins = build_production_plugins()
    assert [plugin.name for plugin in plugins] == [
        "rate_limiter",
        "input_guardrail",
        "output_guardrail",
    ]


def test_assignment_suite_writes_schema_valid_artifacts_without_live_calls(
    monkeypatch, tmp_path
):
    from google.genai import types
    import agents.agent as agent_module
    import core.utils as utils_module
    from assignment import pipeline as pipeline_module
    from assignment.pipeline import build_observability, build_production_plugins
    import jsonschema

    def fake_create_blue_agent(plugins):
        return object(), SimpleNamespace(plugins=plugins)

    async def fake_chat_with_agent(agent, runner, text):
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        )
        for plugin in runner.plugins:
            callback = getattr(plugin, "on_user_message_callback", None)
            if callback is None:
                continue
            blocked = await callback(
                invocation_context=SimpleNamespace(user_id=runner.user_id),
                user_message=user_content,
            )
            if blocked is not None:
                return "".join(part.text for part in blocked.parts), None

        response = SimpleNamespace(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="VinBank banking guidance.")],
            )
        )
        for plugin in runner.plugins:
            callback = getattr(plugin, "after_model_callback", None)
            if callback is not None:
                await callback(callback_context=object(), llm_response=response)
        return "".join(part.text for part in response.content.parts), None

    repo_root = tmp_path / "repo"
    monkeypatch.setattr(
        pipeline_module,
        "__file__",
        str(repo_root / "src" / "assignment" / "pipeline.py"),
    )
    monkeypatch.setattr(agent_module, "create_blue_agent", fake_create_blue_agent)
    monkeypatch.setattr(utils_module, "chat_with_agent", fake_chat_with_agent)
    audit, monitor = build_observability()
    result = asyncio.run(
        pipeline_module.run_assignment_suite(
            {
                "plugins": build_production_plugins(),
                "audit": audit,
                "monitor": monitor,
            }
        )
    )

    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "results.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)
    assert len(result["safe_queries"]) == 5
    assert all(not query["blocked"] for query in result["safe_queries"])
    assert len(result["attack_queries"]) == 7
    assert sum(query["blocked"] for query in result["attack_queries"]) >= 5
    assert all(query["layer"] == "input_guardrail" for query in result["attack_queries"])
    assert len(result["edge_cases"]) == 3
    assert all(query["layer"] == "input_guardrail" for query in result["edge_cases"])
    assert result["rate_limit"] == {
        "max_requests": 10,
        "window_seconds": 60,
        "sent": 15,
        "passed": 10,
        "blocked": 5,
    }
    assert (repo_root / "outputs" / "audit_log.json").is_file()
    assert (repo_root / "outputs" / "metrics.json").is_file()
    assert len(audit.logs) == 30
