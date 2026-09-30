"""Project-scoped exact search reuse and validated local source hydration."""

from datetime import datetime, timezone, timedelta
from researchpilot.paper_candidate import PaperCandidate, QueryHit
from researchpilot.project_models import ProjectQuery, identity, text_key


class ProjectSearchSession:
    """Reuse only identical request meaning, filters and page size for at most one day."""
    def __init__(self, backend, store, request, run_id):
        self.backend, self.store, self.request, self.run_id = backend, store, request, run_id
        self.records, self.reused = {}, 0
        self.scope = identity("scope", type(request).__name__, text_key(getattr(request, "question", None) or request.claim),
            getattr(request, "field", None), getattr(request, "context", None), request.year_from, request.year_to)

    def search_candidates(self, queries, per_query=10, year_from=None, year_to=None):
        found = {}
        for query in queries:
            key = identity("query", self.scope, text_key(query.text), per_query, year_from, year_to)
            prior = None
            if not self.request.refresh_search:
                try:
                    prior = self.store.get_record(self.request.project_id, "queries", key)
                except ValueError:
                    pass
                if prior and datetime.now(timezone.utc) - datetime.fromisoformat(prior.updated_at) > timedelta(days=1):
                    prior = None
            if prior:
                rows = prior.candidates
                self.reused += 1
            else:
                rows = self.backend.search_candidates([query], per_query=per_query, year_from=year_from, year_to=year_to)
            rows = [PaperCandidate.model_validate(row.model_dump()) for row in rows]
            remapped = [PaperCandidate(paper=row.paper, hits=[QueryHit(query_id=query.query_id,
                        query_text=query.text, rank=min(h.rank for h in row.hits))]) for row in rows if row.hits]
            if not prior:
                self.records[key] = ProjectQuery(project_id=self.request.project_id, query_id=key,
                    query=query.text, request_scope=self.scope, per_query=per_query, candidates=remapped,
                    originating_run_id=self.run_id)
            for row in remapped:
                old = found.get(row.paper.paper_id)
                if old is None:
                    found[row.paper.paper_id] = row
                else:
                    hits = {h.query_id: h for h in old.hits}
                    for hit in row.hits:
                        if hit.query_id not in hits or hit.rank < hits[hit.query_id].rank:
                            hits[hit.query_id] = hit
                    found[row.paper.paper_id] = PaperCandidate(paper=old.paper, hits=list(hits.values()))
        return list(found.values())

    def previous_query_texts(self):
        """Local duplicate guard for supplementary recovery; no extra model context."""
        return [row.query for row in self.store.list_records(self.request.project_id, "queries")]


def seed_project(context, *, max_papers):
    """Never seed prior claims as already verified; current claims start empty."""
    from researchpilot.research_models import SelectedPaper
    from researchpilot.research_iteration import EvidenceGapRecord
    if context is None:
        return [], {}, {}, []
    papers = [SelectedPaper(citation_label=f"P{i}", group_id=row.version_group_id, paper=row.paper.model_copy(deep=True))
              for i, row in enumerate(context.papers[:max_papers], 1)]
    ids = {p.paper.paper_id for p in papers}
    titles = {p.paper.paper_id: p.paper.title for p in papers}
    # Metadata can evolve for one canonical ID. Original records remain intact;
    # a current run uses one consistent display title for that source.
    passages = {e.passage.passage_id: e.passage.model_copy(deep=True, update={"title": titles[e.passage.paper_id]})
                for e in context.evidence if e.passage.paper_id in ids}
    visuals = {e.passage.passage_id: e.visual_record.model_copy(deep=True) for e in context.evidence
               if e.visual_record and e.passage.passage_id in passages}
    gaps = [EvidenceGapRecord(gap_id=g.gap_id, description=g.description, search_focus=g.search_focus,
        related_claim_ids=[], severity=g.severity, first_seen_round=1, last_updated_round=1,
        status="unresolved", resolution_reason=g.resolution_reason) for g in context.gaps]
    return papers, passages, visuals, gaps
