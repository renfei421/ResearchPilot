"""Bounded, cached rendering of exactly one physical PDF page. No OCR."""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path

from pydantic import Field
import pymupdf

from researchpilot.document_acquisition import MAX_PDF_BYTES, _atomic_write
from researchpilot.evidence import EvidenceModel


class RenderConfig(EvidenceModel):
    dpi: int = Field(default=160, strict=True, ge=96, le=220)
    max_dimension: int = Field(default=2400, strict=True, ge=256, le=3000)
    max_pixels: int = Field(default=4_000_000, strict=True, ge=65536, le=6_000_000)
    max_png_bytes: int = Field(default=4 * 1024 * 1024, strict=True, ge=1024, le=5 * 1024 * 1024)


@dataclass(frozen=True)
class RenderedPage:
    paper_id: str
    page_number: int
    png: bytes
    pdf_sha256: str
    image_sha256: str
    render_version: str
    width: int
    height: int
    cache_hit: bool = False


class PDFPageRenderer:
    def __init__(self, cache_dir: Path | str, config: RenderConfig | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config or RenderConfig()
        self.version = f"target-page-v1/pymupdf-{pymupdf.VersionBind}/" + sha256(
            self.config.model_dump_json().encode()).hexdigest()[:16]

    def render(self, pdf: Path | bytes, *, paper_id: str, page_number: int) -> RenderedPage:
        if type(page_number) is not int or page_number < 1:
            raise ValueError("page_number must be a positive one-based physical PDF page.")
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError("paper_id is required.")
        if isinstance(pdf, Path):
            if pdf.stat().st_size > MAX_PDF_BYTES:
                raise ValueError("PDF exceeds the rendering byte limit.")
            data = pdf.read_bytes()
        else:
            data = pdf
        if not isinstance(data, bytes) or len(data) > MAX_PDF_BYTES or not data.startswith(b"%PDF-"):
            raise ValueError("Rendering requires bounded PDF bytes or a PDF Path.")
        digest = sha256(data).hexdigest()
        key = sha256(json.dumps([digest, page_number, self.version]).encode()).hexdigest()
        path, manifest = self.cache_dir / f"{key}.png", self.cache_dir / f"{key}.json"
        try:
            document = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise ValueError("Cannot open target PDF for rendering.") from exc
        with document:
            if document.is_encrypted or not 1 <= document.page_count <= 500:
                raise ValueError("Encrypted or oversized PDF cannot be rendered.")
            if page_number > document.page_count:
                raise ValueError("Target page_number is outside the physical PDF page range.")
            try:
                meta = json.loads(manifest.read_text(encoding="utf-8"))
                if path.stat().st_size <= self.config.max_png_bytes:
                    png = path.read_bytes()
                    if meta["image_sha256"] == sha256(png).hexdigest():
                        pix = pymupdf.Pixmap(png)
                        if (meta["pdf_sha256"], meta["page_number"], meta["render_version"]) == (digest, page_number, self.version):
                            self._check_size(pix.width, pix.height, png)
                            return RenderedPage(paper_id, page_number, png, digest, meta["image_sha256"],
                                                self.version, pix.width, pix.height, True)
            except (OSError, ValueError, KeyError, RuntimeError):
                pass  # Corrupt/stale cache is never evidence; render the original page.
            page = document.load_page(page_number - 1)
            width, height = page.rect.width, page.rect.height
            if width <= 0 or height <= 0:
                raise ValueError("Invalid PDF page dimensions.")
            # Leave a pixel margin for MuPDF's outward rounding.
            scale = min(self.config.dpi / 72, (self.config.max_dimension - 2) / max(width, height),
                        sqrt(self.config.max_pixels / ((width + 2) * (height + 2))))
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB,
                                  alpha=False, annots=False)
            png = pix.tobytes("png")
            self._check_size(pix.width, pix.height, png)
            image_hash = sha256(png).hexdigest()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(path, png)
            _atomic_write(manifest, json.dumps(dict(pdf_sha256=digest, page_number=page_number,
                render_version=self.version, image_sha256=image_hash)).encode())
            return RenderedPage(paper_id, page_number, png, digest, image_hash,
                                self.version, pix.width, pix.height)

    def _check_size(self, width: int, height: int, png: bytes) -> None:
        if (max(width, height) > self.config.max_dimension or width * height > self.config.max_pixels
                or len(png) > self.config.max_png_bytes):
            raise ValueError("Target page image exceeds the configured size limit.")
