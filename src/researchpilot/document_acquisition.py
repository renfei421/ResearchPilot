"""Bounded PDF acquisition, page extraction, and explicit abstract fallback."""

from contextlib import nullcontext
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
from pypdf import PdfReader, __version__ as pypdf_version

from researchpilot.evidence import AcquiredDocument, DocumentPage
from researchpilot.paper import Paper


MAX_PDF_BYTES = 25 * 1024 * 1024
_CACHE_VERSION = f"pypdf-{pypdf_version}-pages-v1"


class PdfTextExtractor:
    """Physical, one-based PDF pages, not the publisher's printed page labels.

    No OCR, layout reconstruction, or claims about formulas/tables. Malformed,
    encrypted, oversized, and textless documents raise ValueError to the caller.
    """

    def extract(self, data: bytes, paper: Paper, source_url: str) -> list[DocumentPage]:
        if not data.startswith(b"%PDF-") or len(data) > MAX_PDF_BYTES:
            raise ValueError("Not a supported PDF or PDF exceeds the size limit.")
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted or len(reader.pages) > 500:
                raise ValueError("Encrypted PDF or PDF exceeds 500 pages.")
            pages = []
            for number, page in enumerate(reader.pages, 1):
                contents = page.get_contents()
                if contents is not None and len(contents.get_data()) > 8 * 1024 * 1024:
                    raise ValueError("PDF page content exceeds the extraction limit.")
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(DocumentPage(
                        paper_id=paper.paper_id, title=paper.title, page_number=number,
                        source_type="pdf", text=text, source_url=source_url,
                    ))
        except Exception as exc:
            raise ValueError("PDF could not be extracted safely.") from exc
        if not pages:
            raise ValueError("PDF has no extractable text; OCR is not available.")
        return pages


def _pdf_url(url: str) -> str:
    """Recognize arXiv's documented abs/pdf URL pair, without calling its API."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username:
        raise ValueError("Open-access URL must be an HTTP(S) URL without credentials.")
    if parts.hostname.lower() in ("arxiv.org", "www.arxiv.org") and parts.path.startswith("/abs/"):
        return urlunsplit(("https", "arxiv.org", "/pdf/" + parts.path[5:], "", ""))
    return url


def _digest_pages(pages: list[dict]) -> str:
    return sha256(json.dumps(pages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def abstract_document(paper: Paper, warnings: list[str] | None = None) -> AcquiredDocument:
    messages = list(warnings or [])
    pages = []
    if paper.abstract and paper.abstract.strip():
        pages.append(DocumentPage(
            paper_id=paper.paper_id, title=paper.title, page_number=None,
            source_type="abstract", text=paper.abstract, source_url=paper.open_access_url,
        ))
        messages.append(f"{paper.paper_id}: using abstract only; full text unavailable.")
    else:
        messages.append(f"{paper.paper_id}: no usable PDF or abstract.")
    return AcquiredDocument(pages=pages, warnings=messages)


class DocumentAcquirer:
    """Try one OA URL, then fall back. Cache stores PDFs and checksummed pages.

    Cache keys include paper identity and URL. Metadata is rebound to the current
    Paper on every load. Corrupt page caches are rebuilt from the cached PDF;
    corrupt PDFs are fetched again. Injected HTTP clients remain caller-owned.
    """

    def __init__(self, cache_dir: Path | str = ".researchpilot_cache/documents", *,
                 http_client: httpx.Client | None = None,
                 extractor: PdfTextExtractor | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self._http_client = http_client
        self._extractor = extractor or PdfTextExtractor()

    def cached_pdf(self, paper: Paper) -> Path | None:
        """Read-only rescue access; never download, re-extract, or repair a cache."""
        if not paper.open_access_url:
            return None
        try:
            url = _pdf_url(paper.open_access_url)
            key = sha256(json.dumps([paper.paper_id, url]).encode()).hexdigest()
            path = self.cache_dir / f"{key}.pdf"
            if path.stat().st_size > MAX_PDF_BYTES:
                return None
            if self._read_pages(self.cache_dir / f"{key}.json", path.read_bytes(), paper, url) is None:
                return None
            return path
        except (OSError, ValueError):
            return None

    def acquire(self, paper: Paper) -> AcquiredDocument:
        warnings: list[str] = []
        if not paper.open_access_url:
            return abstract_document(paper)
        try:
            url = _pdf_url(paper.open_access_url)
            key = sha256(json.dumps([paper.paper_id, url]).encode()).hexdigest()
            pdf_path = self.cache_dir / f"{key}.pdf"
            text_path = self.cache_dir / f"{key}.json"
            if pdf_path.is_file():
                try:
                    if pdf_path.stat().st_size > MAX_PDF_BYTES:
                        raise ValueError("Oversized cached PDF.")
                    data = pdf_path.read_bytes()
                    pages = self._read_pages(text_path, data, paper, url)
                    if pages is None:
                        warnings.append(f"{paper.paper_id}: rebuilding missing/invalid page cache.")
                        pages = self._extractor.extract(data, paper, url)
                        self._save_pages(text_path, data, pages, warnings)
                    return AcquiredDocument(pages=pages, warnings=warnings, cache_hit=True)
                except Exception as exc:
                    warnings.append(f"{paper.paper_id}: invalid PDF cache ({type(exc).__name__}); fetching again.")
            data = self._download(url)
            pages = self._extractor.extract(data, paper, url)
            try:
                _atomic_write(pdf_path, data)
                self._save_pages(text_path, data, pages, warnings)
            except OSError:
                warnings.append(f"{paper.paper_id}: PDF cache write failed; using downloaded text.")
            return AcquiredDocument(pages=pages, warnings=warnings)
        except Exception as exc:
            # Do not expose request URLs/query secrets from HTTP exception strings.
            warnings.append(f"{paper.paper_id}: PDF acquisition failed ({type(exc).__name__}).")
            return abstract_document(paper, warnings)

    def _download(self, url: str) -> bytes:
        context = nullcontext(self._http_client) if self._http_client is not None else httpx.Client()
        started = monotonic()
        with context as client:
            with client.stream("GET", url, timeout=15.0, follow_redirects=True,
                               headers={"Accept": "application/pdf", "User-Agent": "ResearchPilot/0.1"}) as response:
                response.raise_for_status()
                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_PDF_BYTES or monotonic() - started > 60:
                        raise ValueError("PDF download exceeded size/time limit.")
                    if len(data) >= 5 and not data.startswith(b"%PDF-"):
                        raise ValueError("OA URL did not return a PDF (no HTML crawling).")
                return bytes(data)

    def _read_pages(self, path: Path, data: bytes, paper: Paper, url: str) -> list[DocumentPage] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["pages"]
            if (payload["version"] != _CACHE_VERSION
                    or payload["pdf_sha256"] != sha256(data).hexdigest()
                    or payload["pages_sha256"] != _digest_pages(rows)):
                return None
            pages = [DocumentPage(paper_id=paper.paper_id, title=paper.title,
                                  source_type="pdf", source_url=url, **row) for row in rows]
            numbers = [p.page_number for p in pages]
            if not pages or numbers != sorted(set(numbers)):
                return None
            return pages
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _save_pages(self, path: Path, data: bytes, pages: list[DocumentPage], warnings: list[str]) -> None:
        rows = [{"page_number": p.page_number, "text": p.text} for p in pages]
        payload = {"version": _CACHE_VERSION, "pdf_sha256": sha256(data).hexdigest(),
                   "pages_sha256": _digest_pages(rows), "pages": rows}
        try:
            _atomic_write(path, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        except OSError:
            warnings.append("Extracted page cache write failed; using in-memory pages.")
