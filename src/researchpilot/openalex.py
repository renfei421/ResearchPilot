"""Pure normalization helpers for OpenAlex Work dictionaries."""

from typing import Any

from researchpilot.paper import Paper


def normalize_doi(doi: str | None) -> str | None:
    """Strip common DOI prefixes and whitespace, then return lowercase text.

    Missing or blank values return None. This does not validate DOI syntax.
    """
    if doi is None:
        return None

    normalized = doi.strip().lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break

    return normalized or None


def reconstruct_abstract(
    abstract_inverted_index: dict[str, list[int]] | None,
) -> str | None:
    """Join words in position order, preserving every repeated occurrence.

    Positions are assumed to be valid and unique across tokens. Missing or
    empty indexes, including indexes without any positions, return None.
    """
    if not abstract_inverted_index:
        return None

    positioned_words = [
        (position, word)
        for word, positions in abstract_inverted_index.items()
        for position in positions
    ]
    positioned_words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned_words) or None


def openalex_work_to_paper(work: dict[str, Any]) -> Paper:
    """Normalize a Work without modifying the input or performing any I/O.

    A non-empty OpenAlex id and a non-blank string title are required. Missing optional
    metadata becomes None, missing authors become [], and missing or null
    citation counts become 0. Author entries without names are skipped.
    """
    source_id = (work.get("id") or "").rstrip("/").rsplit("/", 1)[-1]
    if not source_id:
        raise ValueError("OpenAlex work must have a non-empty id.")

    title = work.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("OpenAlex work title must be a non-empty string.")

    doi = normalize_doi(work.get("doi"))
    authors = []
    for authorship in work.get("authorships") or []:
        name = (authorship.get("author") or {}).get("display_name")
        if name:
            authors.append(name)

    return Paper(
        paper_id=f"doi:{doi}" if doi else f"openalex:{source_id}",
        title=title,
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        authors=authors,
        publication_year=work.get("publication_year"),
        doi=doi,
        citation_count=work.get("cited_by_count") or 0,
        source="openalex",
        source_id=source_id,
        open_access_url=(work.get("open_access") or {}).get("oa_url"),
        relevance_score=None,
    )
