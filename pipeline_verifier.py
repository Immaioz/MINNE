"""Citation verification helpers for the research pipeline."""

from __future__ import annotations

import asyncio
import re
from urllib.parse import quote

import httpx


class CitationVerifier:
    """L1 (arXiv API) + L2 (DOI resolution via Crossref) verification."""

    ARXIV_URL = "http://export.arxiv.org/api/query"
    CROSSREF_URL = "https://api.crossref.org/works"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def close(self) -> None:
        await self._client.aclose()

    async def verify_arxiv(self, title: str) -> tuple[bool, str]:
        """Cerca il titolo su arXiv. Restituisce (passato, messaggio)."""
        query = quote(title)
        url = f"{self.ARXIV_URL}?search_query=ti:{query}&max_results=3"
        await asyncio.sleep(4)
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return False, f"arXiv API error {resp.status_code}"

            import xml.etree.ElementTree as ET

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", ns)
            if not entries:
                return False, "Nessun risultato su arXiv"

            import difflib

            for entry in entries:
                etitle_el = entry.find("atom:title", ns)
                if etitle_el is not None and etitle_el.text:
                    etitle = etitle_el.text.strip().lower()
                    score = difflib.SequenceMatcher(
                        None, title.lower(), etitle,
                    ).ratio()
                    if score > 0.6:
                        return True, f"arXiv match (score={score:.2f})"
            return False, "Titolo non corrisponde su arXiv (arXiv titles differ)"
        except Exception as e:
            return False, f"arXiv error: {e}"

    async def verify_arxiv_id(self, arxiv_id: str) -> tuple[bool, str]:
        """Verifica che un arXiv ID esista."""
        if not arxiv_id:
            return False, "arXiv ID mancante"
        url = f"https://export.arxiv.org/api/query?id_list={quote(arxiv_id)}"
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return False, f"arXiv API error {resp.status_code}"
            import xml.etree.ElementTree as ET

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", ns)
            if entries:
                return True, f"arXiv ID {arxiv_id} valido"
            return False, f"arXiv ID {arxiv_id} non trovato"
        except Exception as e:
            return False, f"arXiv ID error: {e}"

    async def fetch_full_text(self, arxiv_id: str) -> str:
        """Scarica il PDF da arXiv ed estrae il testo completo con PyMuPDF."""
        if not arxiv_id:
            return ""
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        try:
            resp = await self._client.get(pdf_url)
            if resp.status_code != 200:
                return ""
            import fitz
            import io

            doc = fitz.open(stream=io.BytesIO(resp.content), filetype="pdf")
            pages = [page.get_text() for page in doc]
            doc.close()
            text = "\n\n".join(pages)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = text.replace("\x00", "")
            return text.strip()
        except Exception:
            return ""

    async def fetch_arxiv_abstract(self, arxiv_id: str) -> str:
        """Recupera l'abstract da arXiv API (fallback se PDF non disponibile)."""
        if not arxiv_id:
            return ""
        url = f"https://export.arxiv.org/api/query?id_list={quote(arxiv_id)}"
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return ""
            import xml.etree.ElementTree as ET

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            entry = root.find("atom:entry", ns)
            if entry is None:
                return ""
            summary_el = entry.find("atom:summary", ns)
            if summary_el is not None and summary_el.text:
                text = re.sub(r"\s+", " ", summary_el.text.strip())
                return text
            return ""
        except Exception:
            return ""

    async def verify_doi(self, doi: str, expected_title: str) -> tuple[bool, str]:
        """Risolve DOI via Crossref API e verifica il titolo."""
        clean_doi = doi.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:").strip()
        if not clean_doi:
            return False, "DOI mancante"

        url = f"{self.CROSSREF_URL}/{quote(clean_doi)}"
        try:
            resp = await self._client.get(url)
            if resp.status_code != 200:
                return False, f"Crossref error {resp.status_code}"

            data = resp.json()
            msg = data.get("message", {})
            titles = msg.get("title", [])
            if not titles:
                return False, "Nessun titolo in Crossref"

            import difflib

            crossref_title = titles[0].strip().lower()
            score = difflib.SequenceMatcher(
                None, expected_title.lower(), crossref_title,
            ).ratio()
            if score > 0.6:
                return True, f"DOI match (score={score:.2f})"
            return False, f"Titolo Crossref non corrisponde (score={score:.2f})"
        except Exception as e:
            return False, f"DOI error: {e}"

    async def verify_article(self, title: str, doi: str, arxiv_id: str = "") -> tuple[bool, str]:
        """Verifica solo tramite DOI (Crossref)."""
        has_doi = doi.lower().strip() not in ("", "doi not confirmed", "not available", "none", "n/a")
        if not has_doi:
            return False, "DOI non disponibile"
        l2_ok, l2_msg = await self.verify_doi(doi, title)
        if l2_ok:
            return True, f"L2 OK ({l2_msg})"
        return False, f"L2: {l2_msg}"
