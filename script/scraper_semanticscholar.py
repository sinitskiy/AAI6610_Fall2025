import os, re, time, textwrap, pathlib, csv
from typing import Dict, List, Optional
import requests
from tqdm import tqdm

# Settings
QUERY = "uncertainty prediction ML"  # adjust your keywords
YEAR_START, YEAR_END = 2020, 2025                  # inclusive filter

# Target counts
TARGET_TOTAL_S2         = 1200   # Semantic Scholar OA PDFs
TARGET_TOTAL_OPENREVIEW = 1200   # OpenReview PDFs
TARGET_TOTAL_UNPAYWALL  = 1200   # Unpaywall PDFs (via S2 DOIs)

BASE_OUT         = pathlib.Path("papers_uncertainty_prediction_ml")
OUT_S2           = BASE_OUT / "semanticscholar"
OUT_OPENREVIEW   = BASE_OUT / "openreview"
OUT_UNPAYWALL    = BASE_OUT / "unpaywall"

# Semantic Scholar paging
S2_PER_PAGE   = 100   
S2_MAX_PAGES  = 200   
S2_FIELDS = ",".join([
    "title","abstract","year","publicationDate",
    "externalIds","openAccessPdf","publicationVenue"
])

# ============ ENV ============
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

UNPAYWALL_EMAIL   = os.environ.get("UNPAYWALL_EMAIL", "").strip()
S2_API_KEY        = os.environ.get("S2_API_KEY", "").strip()
OPENREVIEW_MAILTO = os.environ.get("OPENREVIEW_MAILTO", "").strip()

# ============ HELPERS ============
def ensure_dirs():
    OUT_S2.mkdir(parents=True, exist_ok=True)
    OUT_OPENREVIEW.mkdir(parents=True, exist_ok=True)
    OUT_UNPAYWALL.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str, limit: int = 180) -> str:
    name = re.sub(r"[\\/*?\"<>|:]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:limit]

def write_sidecar_txt(pdf_path: pathlib.Path, title: str, abstract: str, date: str, doi: str):
    txt = [
        f"Title: {title or ''}",
        f"Date: {date or ''}",
        f"DOI: {doi or ''}",
        "",
        "Abstract:",
        textwrap.fill(abstract or "", width=100)
    ]
    pdf_path.with_suffix(".txt").write_text("\n".join(txt), encoding="utf-8")

def backoff_sleep(i: int):
    time.sleep(min(2 ** i, 8))

def get_json(url: str, params: dict = None, headers: dict = None,
             timeout: int = 30, tries: int = 4) -> Optional[dict]:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 502, 503, 504):
                backoff_sleep(i + 1); continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1: return None
            backoff_sleep(i + 1)
    return None

def download_pdf(url: str, dest: pathlib.Path, tries: int = 3) -> bool:
    for i in range(tries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                if r.status_code in (429, 502, 503, 504):
                    backoff_sleep(i + 1); continue
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with open(dest, "wb") as f, tqdm(
                    total=total if total else None, unit="B", unit_scale=True, desc=dest.name
                ) as bar:
                    for chunk in r.iter_content(1024 * 64):
                        if chunk:
                            f.write(chunk)
                            if total: bar.update(len(chunk))
            return True
        except Exception:
            if i == tries - 1: return False
            backoff_sleep(i + 1)
    return False

def parse_year_from_any(s: Optional[str]) -> Optional[int]:
    if not s: return None
    m = re.search(r"\b(\d{4})\b", s)
    return int(m.group(1)) if m else None

def in_year_window(publicationDate: Optional[str], year: Optional[int]) -> bool:
    y = year if isinstance(year, int) else parse_year_from_any(publicationDate)
    return (y is not None) and (YEAR_START <= y <= YEAR_END)

def norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

# Semantic scholar
def s2_headers() -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if S2_API_KEY: h["x-api-key"] = S2_API_KEY
    return h

def s2_search_paginated(query: str, per_page: int, max_pages: int) -> List[dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    out: List[dict] = []
    offset = 0
    for _ in range(max_pages):
        params = {"query": query, "offset": offset, "limit": per_page, "fields": S2_FIELDS}
        data = get_json(url, params=params, headers=s2_headers(), timeout=40, tries=4)
        if not data: break
        rows = data.get("data", [])
        if not rows: break
        out.extend(rows)
        if len(rows) < per_page: break
        offset += per_page
        time.sleep(0.3)
    return out

def pipeline_semanticscholar(query: str, target_total: int) -> List[dict]:
    """
    Download ONLY from S2 openAccessPdf into OUT_S2.
    Returns list of downloaded records.
    """
    raw = s2_search_paginated(query, S2_PER_PAGE, S2_MAX_PAGES)
    downloaded = []
    seen_titles = set()
    for p in raw:
        if len(downloaded) >= target_total: break
        if not in_year_window(p.get("publicationDate"), p.get("year")):
            continue
        title = p.get("title") or ""
        if not title: continue
        tkey = norm_title(title)
        if tkey in seen_titles: continue
        seen_titles.add(tkey)

        pdf_url = (p.get("openAccessPdf") or {}).get("url")
        if not pdf_url:
            continue

        pdf_path = OUT_S2 / (sanitize_filename(title) + ".pdf")
        if download_pdf(pdf_url, pdf_path):
            write_sidecar_txt(
                pdf_path, title,
                p.get("abstract","") or "",
                p.get("publicationDate") or (str(p.get("year")) if p.get("year") else ""),
                (p.get("externalIds") or {}).get("DOI", "") or ""
            )
            downloaded.append({
                "title": title,
                "abstract": p.get("abstract","") or "",
                "date": p.get("publicationDate") or (str(p.get("year")) if p.get("year") else ""),
                "doi": (p.get("externalIds") or {}).get("DOI", "") or "",
                "pdf_path": str(pdf_path),
                "txt_path": str(pdf_path.with_suffix(".txt")),
            })
    return downloaded

def s2_iter_candidates_for_unpaywall(query: str, need: int, exclude_titles: set, exclude_dois: set) -> List[dict]:
    """
    Discover candidates via S2 (filter years), IGNORE S2 PDFs, keep those with DOIs for Unpaywall.
    """
    raw = s2_search_paginated(query, S2_PER_PAGE, S2_MAX_PAGES)
    candidates = []
    seen = set()
    for p in raw:
        if len(candidates) >= need * 2:
            break
        if not in_year_window(p.get("publicationDate"), p.get("year")):
            continue
        title = p.get("title") or ""
        tkey = norm_title(title)
        doi = (p.get("externalIds") or {}).get("DOI")
        doi_low = (doi or "").lower()
        if not title or not doi or tkey in exclude_titles or (doi_low and doi_low in exclude_dois):
            continue
        if doi_low in seen:
            continue
        seen.add(doi_low)
        candidates.append({
            "title": title,
            "abstract": p.get("abstract","") or "",
            "date": p.get("publicationDate") or (str(p.get("year")) if p.get("year") else ""),
            "doi": doi
        })
    return candidates

# Openreview
def openreview_search_paginated(query: str, limit_total: int) -> List[dict]:
    url = "https://api.openreview.net/notes"
    out: List[dict] = []
    per_page = 100
    offset = 0
    while len(out) < limit_total:
        params = {"content.title": query, "limit": per_page, "offset": offset}
        if OPENREVIEW_MAILTO:
            params["mailto"] = OPENREVIEW_MAILTO
        data = get_json(url, params=params, timeout=30, tries=4)
        if not data or not data.get("notes"):
            break
        batch = []
        for n in data["notes"]:
            c = n.get("content", {})
            title = (c.get("title") or {}).get("value", "") if isinstance(c.get("title"), dict) else c.get("title", "")
            abstract = (c.get("abstract") or {}).get("value", "") if isinstance(c.get("abstract"), dict) else c.get("abstract", "")
            year = (c.get("year") or {}).get("value", "") if isinstance(c.get("year"), dict) else c.get("year", "")
            if not title or not str(year).isdigit():
                continue
            y = int(year)
            if not (YEAR_START <= y <= YEAR_END):
                continue
            note_id = n.get("id")
            if not note_id:
                continue
            batch.append({
                "title": title,
                "abstract": abstract or "",
                "date": str(y),
                "doi": "",
                "pdf_url": f"https://openreview.net/pdf?id={note_id}"
            })
        out.extend(batch)
        if len(batch) < per_page:
            break
        offset += per_page
        time.sleep(0.3)
    return out[:limit_total]

def pipeline_openreview(query: str, target_total: int) -> List[dict]:
    items = openreview_search_paginated(query, target_total)
    downloaded = []
    seen = set()
    for r in items:
        tkey = norm_title(r["title"])
        if tkey in seen:
            continue
        seen.add(tkey)
        pdf_url = r.get("pdf_url")
        if not pdf_url:
            continue
        pdf_path = OUT_OPENREVIEW / (sanitize_filename(r["title"]) + ".pdf")
        if download_pdf(pdf_url, pdf_path):
            write_sidecar_txt(pdf_path, r["title"], r.get("abstract",""), r.get("date",""), r.get("doi",""))
            downloaded.append({**r, "pdf_path": str(pdf_path), "txt_path": str(pdf_path.with_suffix(".txt"))})
    return downloaded

# Unpaywall
def unpaywall_pdf_url(doi: Optional[str]) -> Optional[str]:
    if not doi or not UNPAYWALL_EMAIL:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}"
    data = get_json(url, params={"email": UNPAYWALL_EMAIL}, timeout=30, tries=4)
    if not data:
        return None
    loc = data.get("best_oa_location") or {}
    return loc.get("url_for_pdf") or None

def pipeline_unpaywall_from_s2(query: str, target_total: int,
                               exclude_titles: set, exclude_dois: set) -> List[dict]:
    """
    Discover via S2 (DOIs), download via Unpaywall into OUT_UNPAYWALL.
    Excludes items already downloaded from S2 or OpenReview by title/DOI.
    """
    candidates = s2_iter_candidates_for_unpaywall(query, target_total, exclude_titles, exclude_dois)
    downloaded = []
    for r in candidates:
        if len(downloaded) >= target_total:
            break
        doi = (r.get("doi") or "").strip()
        pdf_url = unpaywall_pdf_url(doi)
        if not pdf_url:
            continue
        pdf_path = OUT_UNPAYWALL / (sanitize_filename(r["title"]) + ".pdf")
        if download_pdf(pdf_url, pdf_path):
            write_sidecar_txt(pdf_path, r["title"], r.get("abstract",""), r.get("date",""), doi)
            downloaded.append({**r, "pdf_path": str(pdf_path), "txt_path": str(pdf_path.with_suffix(".txt"))})
    return downloaded

def main():
    ensure_dirs()
    print(f"Query: {QUERY}  | Years: {YEAR_START}-{YEAR_END}")
    print(f"Output folders:\n - S2 OA     → {OUT_S2.resolve()}\n - OpenReview → {OUT_OPENREVIEW.resolve()}\n - Unpaywall  → {OUT_UNPAYWALL.resolve()}")

    # Semantic Scholar
    print("\n[1/3] Semantic Scholar (openAccessPdf only)...")
    dl_s2 = pipeline_semanticscholar(QUERY, TARGET_TOTAL_S2)
    print(f"S2 downloaded: {len(dl_s2)}")

    # Prepare exclusion sets
    ex_titles = {norm_title(x["title"]) for x in dl_s2}
    ex_dois   = {(x.get("doi") or "").lower() for x in dl_s2 if x.get("doi")}

    # OpenReview
    print("\n[2/3] OpenReview (direct PDFs)...")
    dl_or = pipeline_openreview(QUERY, TARGET_TOTAL_OPENREVIEW)
    print(f"OpenReview downloaded: {len(dl_or)}")
    ex_titles.update(norm_title(x["title"]) for x in dl_or)
    ex_dois.update((x.get("doi") or "").lower() for x in dl_or if x.get("doi"))

    # Unpaywall
    print("\n[3/3] Unpaywall (via S2 DOIs)...")
    if not UNPAYWALL_EMAIL:
        print("WARNING: UNPAYWALL_EMAIL not set; Unpaywall step will likely return 0.")
    dl_upw = pipeline_unpaywall_from_s2(QUERY, TARGET_TOTAL_UNPAYWALL, ex_titles, ex_dois)
    print(f"Unpaywall downloaded: {len(dl_upw)}")

    # Combined report
    report = BASE_OUT / "download_report_sources.csv"
    with open(report, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title","date","doi","source","pdf_path","txt_path"])
        writer.writeheader()
        for r in dl_s2:
            writer.writerow({"title": r["title"], "date": r.get("date",""), "doi": r.get("doi","") or "", "source": "SemanticScholar", "pdf_path": r["pdf_path"], "txt_path": r["txt_path"]})
        for r in dl_or:
            writer.writerow({"title": r["title"], "date": r.get("date",""), "doi": r.get("doi","") or "", "source": "OpenReview",      "pdf_path": r["pdf_path"], "txt_path": r["txt_path"]})
        for r in dl_upw:
            writer.writerow({"title": r["title"], "date": r.get("date",""), "doi": r.get("doi","") or "", "source": "Unpaywall",       "pdf_path": r["pdf_path"], "txt_path": r["txt_path"]})

    print("\n==== Summary ====")
    print(f"Semantic Scholar PDFs : {len(dl_s2)} → {OUT_S2.resolve()}")
    print(f"OpenReview PDFs       : {len(dl_or)} → {OUT_OPENREVIEW.resolve()}")
    print(f"Unpaywall PDFs        : {len(dl_upw)} → {OUT_UNPAYWALL.resolve()}")
    print(f"Report                : {report.resolve()}")

if __name__ == "__main__":
    main()