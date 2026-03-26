"""Hansard XML / ZIP parsing utilities.

Handles two input formats:

Draft Hansard v3 ZIP (primary — used by the bulk data release)
    Each ZIP contains one XML conforming to Draft_Hansard_v3.xsd.
    Speech text lives in ``<membercontribution>`` elements.
    Date context comes from ``<date format="YYYY-MM-DD">`` elements that
    appear in document order and apply to all contributions that follow
    until the next ``<date>`` element.

Legacy XML schema (kept for forward compatibility)
    pre-1909: speeches in ``<speech>`` tags, text directly inside.
    post-1909: ``<p>`` tags nested within ``<speech>`` tags.

Encoding issues (occasional Latin-1 bytes in otherwise UTF-8 XML) are
handled gracefully via a replacement-character fallback.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Literal

from lxml import etree

logger = logging.getLogger(__name__)

SchemaType = Literal["pre1909", "post1909"]


# ---------------------------------------------------------------------------
# Low-level XML parsing helpers
# ---------------------------------------------------------------------------

def _parse_xml_bytes(raw: bytes, source_name: str = "") -> etree._Element:
    """Parse XML from raw bytes, falling back to Latin-1 on encoding errors.

    Parameters
    ----------
    raw:
        Raw bytes of the XML document.
    source_name:
        Filename or label used only for log messages.

    Returns
    -------
    lxml.etree._Element
        Root element of the parsed tree.
    """
    # Strip a byte-order mark if present
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return etree.fromstring(raw)
    except etree.XMLSyntaxError as exc:
        logger.warning(
            "XML syntax error in %s: %s — retrying with encoding fallback.",
            source_name, exc,
        )
    # Force-declare UTF-8 encoding and replace bad bytes
    raw = re.sub(rb"encoding=[\"'][^\"']+[\"']", b'encoding="utf-8"', raw, count=1)
    text = raw.decode("utf-8", errors="replace")
    try:
        return etree.fromstring(text.encode("utf-8"))
    except etree.XMLSyntaxError as exc2:
        logger.error("Could not parse %s after fallback: %s", source_name, exc2)
        raise


def _parse_xml_safe(path: Path) -> etree._Element:
    """Parse a plain XML file from disk.

    Parameters
    ----------
    path:
        Path to the XML file.
    """
    return _parse_xml_bytes(path.read_bytes(), source_name=path.name)


# ---------------------------------------------------------------------------
# Draft Hansard v3 ZIP schema  (primary path)
# ---------------------------------------------------------------------------

def _itertext_clean(el: etree._Element) -> str:
    """Return all text content of *el* and its descendants, space-joined."""
    parts = []
    for node in el.iter():
        if node.text:
            parts.append(node.text.strip())
        if node.tail:
            parts.append(node.tail.strip())
    return " ".join(p for p in parts if p)


def _extract_from_zip_xml(
    root: etree._Element,
    zip_stem: str,
) -> list[dict]:
    """Extract speeches from a Draft_Hansard_v3 XML root element.

    Iterates the document in tree order, tracking the current date whenever
    a ``<date format="YYYY-MM-DD">`` element is encountered, then yielding
    one record per ``<membercontribution>`` element.

    Parameters
    ----------
    root:
        Root element of a parsed Draft_Hansard_v3 document.
    zip_stem:
        Base filename (without extension) used to build speech IDs.

    Returns
    -------
    list of dict
        Each dict: ``speech_id``, ``year``, ``text``.
    """
    current_year: int | None = None
    speeches: list[dict] = []
    mc_idx = 0

    for el in root.iter():
        tag = el.tag

        if tag == "date":
            fmt = el.get("format", "")
            m = re.match(r"(\d{4})", fmt)
            if m:
                current_year = int(m.group(1))

        elif tag == "membercontribution":
            if current_year is None:
                # Fallback: try to read year from the filename stem
                m2 = re.search(r"\b(\d{4})\b", zip_stem)
                if m2:
                    current_year = int(m2.group(1))
                else:
                    logger.debug(
                        "No date context yet for contribution in %s; skipping.", zip_stem
                    )
                    mc_idx += 1
                    continue

            text = _itertext_clean(el).strip()
            if not text:
                mc_idx += 1
                continue

            speech_id = f"{zip_stem}_mc{mc_idx}"
            speeches.append({"speech_id": speech_id, "year": current_year, "text": text})
            mc_idx += 1

    return speeches


def parse_zip(path: Path) -> list[dict]:
    """Parse a single Draft_Hansard_v3 ZIP archive.

    Expects one XML file inside the ZIP.

    Parameters
    ----------
    path:
        Path to the ``.zip`` file.

    Returns
    -------
    list of dict
        Speeches extracted, each with ``speech_id``, ``year``, ``text``.
    """
    stem = path.stem  # e.g. "S5CV0618P0"
    with zipfile.ZipFile(path, "r") as zf:
        xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
        if not xml_names:
            logger.warning("No XML found inside %s; skipping.", path.name)
            return []
        raw = zf.read(xml_names[0])

    root = _parse_xml_bytes(raw, source_name=path.name)
    return _extract_from_zip_xml(root, zip_stem=stem)


# ---------------------------------------------------------------------------
# Legacy XML schema  (kept for forward-compatibility with old raw XML files)
# ---------------------------------------------------------------------------

def detect_schema(xml_root: etree._Element) -> SchemaType:
    """Detect whether an XML root belongs to the pre-1909 or post-1909 schema.

    Parameters
    ----------
    xml_root:
        Root element of a parsed Hansard XML document.

    Returns
    -------
    SchemaType
        ``"pre1909"`` or ``"post1909"``.
    """
    for speech in xml_root.iter("speech"):
        if speech.find("p") is not None:
            return "post1909"
        date_attr = speech.get("date") or xml_root.get("date") or ""
        if date_attr:
            m = re.match(r"(\d{4})", date_attr)
            if m and int(m.group(1)) >= 1909:
                return "post1909"
    return "pre1909"


def _extract_year_from_root(xml_root: etree._Element) -> int | None:
    """Try to extract a year integer from XML metadata attributes."""
    for attr in ("date", "year"):
        val = xml_root.get(attr, "")
        m = re.search(r"\b(\d{4})\b", val)
        if m:
            return int(m.group(1))
    for child in xml_root:
        for attr in ("date", "year"):
            val = child.get(attr, "")
            m = re.search(r"\b(\d{4})\b", val)
            if m:
                return int(m.group(1))
    return None


def extract_speeches(
    xml_root: etree._Element,
    schema: SchemaType,
    filename: str = "",
) -> list[dict]:
    """Extract speeches from a legacy Hansard XML document.

    Parameters
    ----------
    xml_root:
        Root element of the parsed Hansard XML tree.
    schema:
        Schema variant as returned by :func:`detect_schema`.
    filename:
        Original filename, used as fallback for year extraction.

    Returns
    -------
    list of dict
        Each dict: ``speech_id``, ``year``, ``text``.
    """
    year: int | None = None
    m = re.search(r"\b(\d{4})\b", filename)
    if m:
        year = int(m.group(1))
    if year is None:
        year = _extract_year_from_root(xml_root)
    if year is None:
        logger.warning("Could not determine year for %s; defaulting to 0.", filename)
        year = 0

    speeches = []
    for idx, speech_el in enumerate(xml_root.iter("speech")):
        speech_id = (
            speech_el.get("id")
            or speech_el.get("speakerid")
            or f"{filename}_{idx}"
        )
        if schema == "post1909":
            parts = [p.text or "" for p in speech_el.findall("p")]
            text = " ".join(p.strip() for p in parts if p.strip())
        else:
            text_parts = []
            if speech_el.text:
                text_parts.append(speech_el.text.strip())
            for child in speech_el:
                if child.text:
                    text_parts.append(child.text.strip())
                if child.tail:
                    text_parts.append(child.tail.strip())
            text = " ".join(p for p in text_parts if p)

        if text:
            speeches.append({"speech_id": speech_id, "year": year, "text": text})

    return speeches


def parse_file(path: Path) -> list[dict]:
    """Parse a legacy Hansard plain XML file.

    Parameters
    ----------
    path:
        Path to the XML file.

    Returns
    -------
    list of dict
        Speeches extracted from the file.
    """
    root = _parse_xml_safe(path)
    schema = detect_schema(root)
    logger.debug("%s → schema=%s", path.name, schema)
    return extract_speeches(root, schema, filename=path.name)
