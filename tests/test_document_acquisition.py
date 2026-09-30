import json
from unittest.mock import Mock

import httpx

from phase5_helpers import OfflineTest, TEXT, paper, pdf_bytes
from researchpilot.document_acquisition import DocumentAcquirer, PdfTextExtractor, _pdf_url


class PdfTests(OfflineTest):
    def test_extracts_real_pdf_text_and_preserves_physical_pages(self):
        result = PdfTextExtractor().extract(pdf_bytes(), paper(), "https://example.org/a.pdf")
        self.assertEqual([p.page_number for p in result], [1, 3])
        self.assertIn("Shared caches", result[0].text)
        self.assertTrue(all(p.source_type == "pdf" and p.paper_id == "openalex:W1" for p in result))

    def test_malformed_pdf_is_clear_failure(self):
        with self.assertRaises(ValueError):
            PdfTextExtractor().extract(b"%PDF-broken", paper(), "https://example.org/a.pdf")

    def test_textless_pdf_is_failure(self):
        with self.assertRaises(ValueError):
            PdfTextExtractor().extract(pdf_bytes(("",)), paper(), "https://example.org/a.pdf")

    def test_non_pdf_is_rejected(self):
        with self.assertRaises(ValueError):
            PdfTextExtractor().extract(b"<html>Landing page</html>", paper(), "https://example.org/a")


class AcquisitionTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.root = self.temporary_directory()
        self.requests = []
        self.data = pdf_bytes()
        def handler(request):
            self.requests.append(request)
            return httpx.Response(200, content=self.data, headers={"content-type": "application/pdf"})
        self.http = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(self.http.close)
        self.extractor = Mock(wraps=PdfTextExtractor())
        self.fetcher = DocumentAcquirer(self.root, http_client=self.http, extractor=self.extractor)

    def test_download_and_both_caches_are_reused(self):
        first = self.fetcher.acquire(paper())
        second = self.fetcher.acquire(paper())
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.pages, second.pages)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.extractor.extract.call_count, 1)
        self.assertEqual(len(list(self.root.glob("*.pdf"))), 1)
        self.assertEqual(len(list(self.root.glob("*.json"))), 1)
        self.assertEqual(self.requests[0].extensions["timeout"]["read"], 15.0)
        self.assertFalse(self.http.is_closed)

    def test_corrupt_text_cache_reparses_without_download(self):
        self.fetcher.acquire(paper())
        next(self.root.glob("*.json")).write_text("broken", encoding="utf-8")
        result = self.fetcher.acquire(paper())
        self.assertTrue(result.cache_hit)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.extractor.extract.call_count, 2)

    def test_tampered_cached_text_checksum_is_rejected(self):
        self.fetcher.acquire(paper())
        path = next(self.root.glob("*.json"))
        content = json.loads(path.read_text(encoding="utf-8"))
        content["pages"][0]["text"] = "invented claim"
        path.write_text(json.dumps(content), encoding="utf-8")
        result = self.fetcher.acquire(paper())
        self.assertIn("Shared caches", result.pages[0].text)
        self.assertEqual(self.extractor.extract.call_count, 2)

    def test_corrupt_pdf_redownloads(self):
        self.fetcher.acquire(paper())
        next(self.root.glob("*.pdf")).write_bytes(b"corrupt")
        result = self.fetcher.acquire(paper())
        self.assertEqual(len(self.requests), 2)
        self.assertFalse(result.cache_hit)
        self.assertTrue(result.pages)

    def test_cache_rebinds_current_metadata(self):
        self.fetcher.acquire(paper())
        result = self.fetcher.acquire(paper(title="Updated title"))
        self.assertEqual(result.pages[0].title, "Updated title")
        self.assertEqual(len(self.requests), 1)

    def test_different_url_has_different_cache_key(self):
        self.fetcher.acquire(paper())
        self.fetcher.acquire(paper(open_access_url="https://example.org/new.pdf"))
        self.assertEqual(len(self.requests), 2)

    def test_abstract_fallback_without_url(self):
        result = self.fetcher.acquire(paper(open_access_url=None))
        self.assertEqual(result.pages[0].source_type, "abstract")
        self.assertIsNone(result.pages[0].page_number)
        self.assertEqual(result.pages[0].text, TEXT)
        self.assertFalse(self.requests)

    def test_http_error_falls_back_and_does_not_expose_request_url(self):
        def fail(request):
            return httpx.Response(403)
        with httpx.Client(transport=httpx.MockTransport(fail)) as client:
            result = DocumentAcquirer(self.root, http_client=client).acquire(paper())
        self.assertEqual(result.pages[0].source_type, "abstract")
        self.assertIn("HTTPStatusError", " ".join(result.warnings))
        self.assertNotIn("https://", " ".join(result.warnings))

    def test_timeout_falls_back(self):
        def fail(request):
            raise httpx.ReadTimeout("timeout", request=request)
        with httpx.Client(transport=httpx.MockTransport(fail)) as client:
            result = DocumentAcquirer(self.root, http_client=client).acquire(paper())
        self.assertEqual(result.pages[0].source_type, "abstract")

    def test_landing_page_or_malformed_pdf_falls_back(self):
        for data in (b"<html>not a PDF</html>", b"%PDF-broken"):
            with self.subTest(data=data):
                self.data = data
                result = self.fetcher.acquire(paper())
                self.assertEqual(result.pages[0].source_type, "abstract")
                self.assertFalse(list(self.root.glob("*.pdf")))

    def test_no_abstract_returns_empty_document(self):
        result = self.fetcher.acquire(paper(open_access_url=None, abstract=None))
        self.assertEqual(result.pages, [])
        self.assertTrue(result.warnings)

    def test_cache_write_failure_keeps_downloaded_pdf_text(self):
        blocked = self.root / "file"
        blocked.write_text("not a directory", encoding="utf-8")
        result = DocumentAcquirer(blocked, http_client=self.http).acquire(paper())
        self.assertEqual(result.pages[0].source_type, "pdf")
        self.assertTrue(result.warnings)

    def test_arxiv_abs_url_is_resolved_without_api(self):
        self.assertEqual(_pdf_url("https://arxiv.org/abs/2401.12345v2"), "https://arxiv.org/pdf/2401.12345v2")

    def test_non_http_url_falls_back_without_network(self):
        result = self.fetcher.acquire(paper(open_access_url="file:///private.pdf"))
        self.assertEqual(result.pages[0].source_type, "abstract")
        self.assertFalse(self.requests)
