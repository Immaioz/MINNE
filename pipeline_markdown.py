"""Markdown table and dataset-link enrichment utilities."""

from __future__ import annotations

from pipeline_agents import DATASET_SEARCH_PROMPT, OpencodeClient


def _table_to_list(md_table: str) -> str:
    """
    Convert a wide Markdown table into a per-row key-value list format
    that renders well in PDF regardless of column count.
    """
    lines = md_table.strip().splitlines()
    if not lines:
        return md_table

    header_idx = None
    delim_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and line.strip().endswith("|"):
            if "---" in line:
                delim_idx = i
            elif header_idx is None and delim_idx is None:
                header_idx = i
            if header_idx is not None and delim_idx is not None:
                break

    if header_idx is None or delim_idx is None:
        return md_table

    headers = [h.strip() for h in lines[header_idx].strip().strip("|").split("|")]
    num_cols = len(headers)
    if num_cols <= 5:
        return md_table

    out_parts: list[str] = []
    for line in lines[delim_idx + 1:]:
        stripped = line.strip()
        if not stripped or not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        while len(cells) < num_cols:
            cells.append("")
        cells = cells[:num_cols]

        if all(c == "" or c == "..." for c in cells):
            continue

        item_parts: list[str] = []
        for h, c in zip(headers, cells):
            if h in ("#", ""):
                continue
            if c and c != "...":
                item_parts.append(f"  - **{h}:** {c}")
        if item_parts:
            out_parts.append("\n".join(item_parts))

    if not out_parts:
        return md_table

    return "\n\n".join(out_parts) + "\n"


def fix_wide_tables(markdown_text: str, max_cols: int = 5) -> str:
    """
    Find all Markdown tables in *markdown_text* and convert those with more
    than *max_cols* columns to a row-by-row key-value list format.
    """
    result: list[str] = []
    in_table = False
    table_lines: list[str] = []

    for line in markdown_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(line)
            in_table = True
        else:
            if in_table:
                table_text = "".join(table_lines)
                result.append(_table_to_list(table_text))
                table_lines = []
                in_table = False
            result.append(line)

    if in_table and table_lines:
        table_text = "".join(table_lines)
        result.append(_table_to_list(table_text))

    return "".join(result)


def _find_public_datasets(table_md: str) -> list[tuple[int, str, str]]:
    """
    Parse the aggregated Markdown table and return a list of
    (row_index, dataset_name, current_cell) tuples for datasets
    marked as public but missing a download link/URL.
    """
    lines = table_md.strip().splitlines()
    if not lines:
        return []

    header_idx = None
    delim_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            if "---" in s:
                delim_idx = i
            elif header_idx is None and delim_idx is None:
                header_idx = i
        if header_idx is not None and delim_idx is not None:
            break

    if header_idx is None or delim_idx is None:
        return []

    headers = [h.strip().lower() for h in lines[header_idx].strip().strip("|").split("|")]

    ds_col = None
    for i, h in enumerate(headers):
        if h == "dataset":
            ds_col = i
            break
    if ds_col is None:
        return []

    needs_link: list[tuple[int, str, str]] = []
    for row_idx, line in enumerate(lines[delim_idx + 1:], start=delim_idx + 1):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) <= ds_col:
            continue
        cell = cells[ds_col]
        cell_lower = cell.lower()
        has_url = "http://" in cell or "https://" in cell or "[link]" in cell.lower() or "](http" in cell
        is_public = "public" in cell_lower
        if is_public and not has_url and "none" not in cell_lower:
            name = cell.split("(")[0].split(",")[0].strip()
            if name and name.lower() not in ("public", "dataset", "yes", "n/a", ""):
                needs_link.append((row_idx, name, cell))
    return needs_link


def _update_dataset_cell(table_md: str, row_idx: int, new_cell: str) -> str:
    """Replace the Dataset cell at *row_idx* with *new_cell* in the table."""
    lines = table_md.splitlines()
    line = lines[row_idx]
    cells = line.strip().strip("|").split("|")

    header_idx = None
    delim_idx = None
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith("|") and s.endswith("|"):
            if "---" in s:
                delim_idx = i
            elif header_idx is None and delim_idx is None:
                header_idx = i
        if header_idx is not None and delim_idx is not None:
            break
    headers = [h.strip().lower() for h in lines[header_idx].strip().strip("|").split("|")]
    ds_col = None
    for i, h in enumerate(headers):
        if h == "dataset":
            ds_col = i
            break
    if ds_col is None:
        return table_md

    cells[ds_col] = f" {new_cell} "
    indent = line[:len(line) - len(line.lstrip())]
    lines[row_idx] = indent + "|" + "|".join(cells) + "|"
    return "\n".join(lines)


async def enrich_dataset_links(table_md: str) -> str:
    """
    Search the web for public dataset names missing download links
    and update the table accordingly.
    """
    needs = _find_public_datasets(table_md)
    if not needs:
        return table_md

    print(f"  🔍  ricerca link per {len(needs)} dataset pubblici...")
    result = table_md
    for row_idx, ds_name, old_cell in needs:
        client = OpencodeClient("researcher")
        search_prompt = f"{DATASET_SEARCH_PROMPT}\n\nDataset name: {ds_name}"
        url = await client.ask(search_prompt)
        url = url.strip()
        if url and url != "NOT FOUND" and not url.startswith("[Error"):
            url = url.rstrip(".,;")
            if url.startswith("http://") or url.startswith("https://"):
                new = f"{old_cell} ([link]({url}))"
                result = _update_dataset_cell(result, row_idx, new)
                print(f"    ✓ {ds_name} → {url[:60]}...")
            else:
                print(f"    - {ds_name}: risultato non valido ({url[:40]}...)")
        else:
            print(f"    - {ds_name}: link non trovato")

    return result
