"""Explicit synthetic local browser acceptance app, never a production mode.

Run from the repository root:
  uv run uvicorn product_acceptance_app:app --app-dir tests --host 127.0.0.1 --port 8765
All research providers are handcrafted fakes; the actual ResearchAgent and graph
still run. Results are stored separately from normal production history.
"""

from pathlib import Path
from time import sleep

from researchpilot.api import create_app
from researchpilot.config import Settings
from test_research_graph import loop_fixture


def fixture_agent(progress):
    fixture = loop_fixture()
    fixture.a.title = "Synthetic fixture: repeated-read shared caches"
    fixture.b.title = "Synthetic fixture: contention-aware cache allocation"
    def report(event):
        progress(event)
        sleep(0.2)  # Makes each real graph stage observable by browser polling.
    fixture.agent._on_progress = report
    class OfflineAcceptanceAgent:
        def run(self, request):
            fixture.planner.plan.return_value = fixture.planner.plan.return_value.model_copy(
                update={"research_question": request.question})
            return fixture.agent.run(request)
    return OfflineAcceptanceAgent()


cache = Path(".researchpilot_cache/product-acceptance")
app = create_app(settings=Settings(cache_dir=cache, run_db=cache / "runs.sqlite3"),
                 agent_factory=fixture_agent)
