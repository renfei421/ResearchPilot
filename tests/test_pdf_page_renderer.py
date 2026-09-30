from hashlib import sha256
from unittest.mock import patch

import pymupdf

from phase5_helpers import OfflineTest, pdf_bytes, paper
from researchpilot.document_acquisition import DocumentAcquirer
from researchpilot.pdf_page_renderer import PDFPageRenderer, RenderConfig


def renderable_pdf(texts=("FIRST PAGE", "SECOND TARGET PAGE", "THIRD PAGE")):
    with pymupdf.open() as doc:
        for text in texts:
            page = doc.new_page(width=612, height=792)
            page.insert_text((50, 92), text)
        return doc.tobytes()


class PDFPageRendererTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.directory = self.temporary_directory()
        self.renderer = PDFPageRenderer(self.directory)
        self.data = renderable_pdf()

    def render(self, number=2, data=None):
        return self.renderer.render(data or self.data, paper_id="paper-a", page_number=number)

    def test_exact_physical_target_page(self):
        result = self.render()
        with pymupdf.open(stream=self.data, filetype="pdf") as doc:
            pix = doc.load_page(1).get_pixmap(matrix=pymupdf.Matrix(160 / 72, 160 / 72),
                                            colorspace=pymupdf.csRGB, alpha=False, annots=False)
        self.assertEqual(sha256(result.png).hexdigest(), sha256(pix.tobytes("png")).hexdigest())
        self.assertEqual(result.page_number, 2)

    def test_page_one_is_not_page_two(self):
        self.assertNotEqual(self.render(1).image_sha256, self.render(2).image_sha256)

    def test_zero_page_rejected(self):
        with self.assertRaisesRegex(ValueError, "one-based"):
            self.render(0)

    def test_beyond_last_page_rejected(self):
        with self.assertRaisesRegex(ValueError, "range"):
            self.render(4)

    def test_bool_and_noninteger_page_rejected(self):
        for value in (True, 1.0, "1", -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.render(value)

    def test_only_one_page_is_loaded_and_rendered(self):
        original = pymupdf.Document.load_page
        calls = []
        def load(doc, number):
            calls.append(number)
            return original(doc, number)
        with patch.object(pymupdf.Document, "load_page", load):
            self.render(3)
        self.assertEqual(calls, [2])

    def test_image_dimensions_and_pixel_cap(self):
        self.renderer = PDFPageRenderer(self.directory, RenderConfig(max_dimension=500, max_pixels=150000))
        result = self.render()
        self.assertLessEqual(max(result.width, result.height), 500)
        self.assertLessEqual(result.width * result.height, 150000)

    def test_payload_limit_rejects_without_second_render(self):
        self.renderer = PDFPageRenderer(self.directory, RenderConfig(max_png_bytes=1024))
        with self.assertRaisesRegex(ValueError, "size limit"):
            self.render()

    def test_corrupt_pdf_is_safe(self):
        with self.assertRaises(ValueError):
            self.render(data=b"%PDF-invalid")

    def test_cached_render_skips_load_page(self):
        first = self.render()
        with patch.object(pymupdf.Document, "load_page", side_effect=AssertionError("Do not render twice")):
            second = self.render()
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.png, second.png)

    def test_corrupt_image_cache_rebuilt(self):
        first = self.render()
        next(self.directory.glob("*.png")).write_bytes(b"not-png")
        result = self.render()
        self.assertFalse(result.cache_hit)
        self.assertEqual(first.png, result.png)

    def test_different_render_config_does_not_reuse_cache(self):
        first = self.render()
        self.renderer = PDFPageRenderer(self.directory, RenderConfig(dpi=120))
        second = self.render()
        self.assertNotEqual(first.render_version, second.render_version)
        self.assertFalse(second.cache_hit)

    def test_path_and_bytes_same_page_provenance(self):
        path = self.directory / "source.pdf"
        path.write_bytes(self.data)
        result = self.render(data=path)
        self.assertEqual(result.pdf_sha256, sha256(self.data).hexdigest())
        self.assertEqual(result.image_sha256, sha256(result.png).hexdigest())
        self.assertEqual(result.paper_id, "paper-a")

    def test_read_only_pdf_provider_never_downloads(self):
        acquirer = DocumentAcquirer(self.directory)
        with patch.object(acquirer, "_download", side_effect=AssertionError("network forbidden")):
            self.assertIsNone(acquirer.cached_pdf(paper()))

    def test_pdf_provider_requires_valid_matching_cache(self):
        acquirer = DocumentAcquirer(self.directory)
        with patch.object(acquirer, "_download", return_value=self.data) as download:
            acquirer.acquire(paper())
            path = acquirer.cached_pdf(paper())
            self.assertIsNotNone(path)
            self.assertEqual(download.call_count, 1)
        path.write_bytes(pdf_bytes(("CHANGED",)))
        self.assertIsNone(acquirer.cached_pdf(paper()))
