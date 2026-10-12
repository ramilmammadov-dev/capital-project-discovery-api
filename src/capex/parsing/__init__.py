"""Document parsing entry point with a content-hash cache (parsed pages are stored for reuse/debugging)."""

from __future__ import annotations

from capex.config import ParsingConfig
from capex.fetch import FetchResult
from capex.models import ParsedDocument
from capex.parsing.html import PARSER_VERSION as HTML_PARSER_VERSION
from capex.parsing.html import parse_html
from capex.parsing.pdf import PARSER_VERSION as PDF_PARSER_VERSION
from capex.parsing.pdf import parse_pdf
from capex.storage.cache import FileCache


class UnsupportedDocumentError(ValueError):
    pass


def parse_document(
    fetched: FetchResult, doc_id: str, cache: FileCache, config: ParsingConfig, title: str = ""
) -> tuple[ParsedDocument, bool]:
    """Parse a fetched document. Returns (document, from_cache)."""
    if not fetched.ok or not fetched.content_hash:
        raise UnsupportedDocumentError(fetched.error or "document was not downloaded")
    version = PDF_PARSER_VERSION if fetched.is_pdf else HTML_PARSER_VERSION
    key = f"{fetched.content_hash}-{version}"
    cached = cache.get_json("parsed", key)
    if cached:
        doc = ParsedDocument.model_validate(cached)
        doc.doc_id = doc_id
        return doc, True
    if fetched.is_pdf:
        path = fetched.path or cache.path("downloads", fetched.content_hash, "")
        doc = parse_pdf(
            path,
            doc_id,
            fetched.content_hash,
            max_pages=config.max_pages,
            workers=config.workers,
            ocr_enabled=config.ocr_enabled,
            ocr_max_pages=config.ocr_max_pages,
            title=title,
        )
    elif fetched.is_html:
        html = (fetched.content or b"").decode("utf-8", errors="replace")
        doc = parse_html(html, doc_id, fetched.content_hash, title=title)
    else:
        raise UnsupportedDocumentError(f"unsupported content type {fetched.content_type!r}")
    cache.put_json("parsed", key, doc.model_dump(mode="json"))
    return doc, False
