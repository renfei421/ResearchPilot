"""CLI for the end-to-end agent; importing this module performs no API calls."""

import argparse
from pathlib import Path
import sys

from pydantic import ValidationError

from researchpilot.document_acquisition import _atomic_write
from researchpilot.config import Settings
from researchpilot.research_agent import ResearchAgent
from researchpilot.research_models import ResearchRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research a question with verified source citations.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--year-to", type=int)
    parser.add_argument("--max-papers", type=int, default=6)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--top-passages", type=int, default=20)
    parser.add_argument("--max-search-rounds", type=int, default=2)
    parser.add_argument("--max-follow-up-queries", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        request = ResearchRequest(question=args.question, year_from=args.year_from, year_to=args.year_to,
                                  max_papers=args.max_papers, max_queries=args.max_queries,
                                  top_passages=args.top_passages, max_search_rounds=args.max_search_rounds,
                                  max_follow_up_queries=args.max_follow_up_queries)
        if not Settings.from_env().openai_configured:
            print("OPENAI_API_KEY is required. Set it in the process environment; no API call was made.", file=sys.stderr)
            return 2
        agent = ResearchAgent(model=args.model, progress=lambda message: print(message, file=sys.stderr, flush=True))
        result = agent.run(request)
        if args.output:
            _atomic_write(args.output, result.model_dump_json(indent=2).encode("utf-8"))
    except ValidationError as exc:
        # Do not echo input values (provider payloads may contain source content).
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                            for e in exc.errors(include_input=False, include_url=False))
        print(f"Validation failed: {details}", file=sys.stderr)
        return 1
    except Exception as exc:
        # HTTP exception strings may embed API keys in request query parameters.
        # Retain a useful stage in stderr progress and a safe error type only.
        print(f"Research run failed ({type(exc).__name__}); no complete output was saved.", file=sys.stderr)
        return 1
    print(result.answer)
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    stats = result.run_stats
    for item in result.round_trace:
        print(f"[Round {item.round}] {item.queries} queries; {item.new_papers} new papers; "
              f"{item.new_evidence} new evidence; {item.supported_claims} supported claims; "
              f"{item.remaining_gaps} remaining gaps")
    print(f"Termination: {result.termination_reason}")
    print(f"\nRun: {stats.selected_papers} papers, {stats.evidence_selected} evidence passages, "
          f"{stats.supported} supported / {stats.partially_supported} partial / "
          f"{stats.unsupported} unsupported / {stats.conflicting} conflicting claims; "
          f"{stats.paper_ranking}; {stats.elapsed_seconds:.1f}s")
    if args.output:
        print(f"\nAudit JSON: {args.output}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
