"""Small synthetic fixtures for agent tests; no evaluation datasets or APIs."""

from io import BytesIO
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from researchpilot.evidence import DocumentPage, EvidencePassage
from researchpilot.paper import Paper
from researchpilot.paper_candidate import PaperCandidate, QueryHit
from researchpilot.query_plan import SearchPlan


QUESTION = "How do shared caches reduce storage latency?"
TEXT = "Shared caches reduce storage latency for repeated reads, subject to capacity constraints."


def paper(key="W1", **changes):
    data = dict(paper_id=f"openalex:{key}", title=f"Shared caches {key}", abstract=TEXT,
                authors=["A. Researcher"], publication_year=2024, doi=None, citation_count=0,
                source="openalex", source_id=key, open_access_url="https://papers.example/study.pdf")
    data.update(changes)
    return Paper(**data)


def candidate(record=None, rank=1, query_id="q1"):
    return PaperCandidate(paper=record or paper(), hits=[QueryHit(query_id=query_id, query_text="shared caches", rank=rank)])


def plan():
    return SearchPlan(research_question=QUESTION, concepts=["shared caches", "storage latency"], queries=[
        dict(query_id=f"q{i}", text=text, role=role, rationale="Distinct academic search facet.")
        for i, (text, role) in enumerate([
            ('"shared caches" AND latency', "core"),
            ('"storage contention"', "facet"),
            ('"cache capacity"', "bridge"),
        ], 1)
    ])


def page(record=None, number=1, text=TEXT):
    record = record or paper()
    return DocumentPage(paper_id=record.paper_id, title=record.title, page_number=number,
                        source_type="pdf" if number is not None else "abstract",
                        text=text, source_url=record.open_access_url)


def passage(key="e1", record=None, number=1, text=TEXT):
    return EvidencePassage(**page(record, number, text).model_dump(), passage_id=key)


def pdf_bytes(texts=(TEXT, "", "Capacity limits require careful cache allocation.")):
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for text in texts:
        pdf_page = writer.add_blank_page(width=612, height=792)
        if text:
            pdf_page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            })
            stream = DecodedStreamObject()
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream.set_data(f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET".encode("ascii"))
            pdf_page[NameObject("/Contents")] = stream
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class OfflineTest(unittest.TestCase):
    def setUp(self):
        for name in ("socket.create_connection", "socket.socket.connect"):
            guard = patch(name, side_effect=AssertionError("Automated agent tests must remain offline"))
            guard.start()
            self.addCleanup(guard.stop)

    def temporary_directory(self):
        root = Path(__file__).resolve().parent / f"phase5-fixture-{uuid4().hex}"
        root.mkdir()
        def cleanup():
            resolved = root.resolve()
            if resolved.parent != Path(__file__).resolve().parent or not resolved.name.startswith("phase5-fixture-"):
                raise ValueError("Unsafe fixture cleanup path.")
            shutil.rmtree(resolved)
        self.addCleanup(cleanup)
        return root
