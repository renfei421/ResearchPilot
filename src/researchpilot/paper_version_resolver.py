"""Conservative version grouping using exact normalized metadata matches."""

from hashlib import sha256
import json
import unicodedata

from researchpilot.paper_candidate import PaperCandidate
from researchpilot.paper_version import PaperVersionGroup


def normalize_title(title: str) -> str:
    """Normalize Unicode, case, punctuation, dashes, and whitespace.

    Unicode punctuation becomes word separators. Words and non-punctuation
    symbols are retained, including correction labels and mathematical symbols.
    """
    normalized = unicodedata.normalize("NFKC", title).casefold()
    separated = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(separated.split())


def normalize_author_signature(authors: list[str]) -> tuple[str, ...]:
    """Normalize names without sorting, removing punctuation, or expanding initials."""
    return tuple(
        " ".join(unicodedata.normalize("NFKC", author).casefold().split())
        for author in authors
    )


def _make_group_id(
    title: str, authors: tuple[str, ...], members: list[PaperCandidate]
) -> str:
    # Member ids distinguish groups split by years or missing metadata. Sorting
    # makes identity independent of member order when group membership is equal.
    identity = [title, authors, sorted(member.paper.paper_id for member in members)]
    serialized = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return f"version:{sha256(serialized.encode('utf-8')).hexdigest()}"


class PaperVersionResolver:
    """Group versions without changing candidates, papers, or retrieval hits."""

    def group_versions(
        self, candidates: list[PaperCandidate]
    ) -> list[PaperVersionGroup]:
        """Join the first compatible group, retaining input order and objects.

        Every pair in a group must have known years at most two years apart;
        matching is not transitive through intermediate publication years.
        Empty normalized titles or absent/blank author names remain singletons.
        Group ids describe current membership, so adding a member changes the id.
        """
        member_groups: list[tuple[str, tuple[str, ...], list[PaperCandidate]]] = []

        for candidate in candidates:
            title = normalize_title(candidate.paper.title)
            authors = normalize_author_signature(candidate.paper.authors)
            year = candidate.paper.publication_year

            if not title or not authors or not all(authors) or year is None:
                member_groups.append((title, authors, [candidate]))
                continue

            for group_title, group_authors, members in member_groups:
                if (
                    title == group_title
                    and authors == group_authors
                    and all(
                        member.paper.publication_year is not None
                        and abs(year - member.paper.publication_year) <= 2
                        for member in members
                    )
                ):
                    members.append(candidate)
                    break
            else:
                member_groups.append((title, authors, [candidate]))

        return [
            PaperVersionGroup(
                group_id=_make_group_id(title, authors, members), members=members
            )
            for title, authors, members in member_groups
        ]
