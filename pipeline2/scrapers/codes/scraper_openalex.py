#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized OpenAlex + OpenReview + Unpaywall Crawler
Improved PDF download success rate, optimized search strategy, added progress display
"""

import os
import re
import time
import json
import csv
import textwrap
from typing import Dict, Any, List, Optional, Set
from urllib.parse import quote_plus
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path

# ========== Path Configuration ==========
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRAPER_OUTPUTS = PROJECT_ROOT / "scrapers" / "outputs"

BASE_OUT = SCRAPER_OUTPUTS / "openalex_openreview_papers"
OUT_OPENALEX = BASE_OUT / "openalex"
OUT_OPENREVIEW = BASE_OUT / "openreview"
OUT_UNPAYWALL = BASE_OUT / "unpaywall"

# ========== Load Environment Variables ==========
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# ========== Configuration ==========
# Extended search terms to improve recall
QUERIES = [
    "uncertainty quantification machine learning",
    "bayesian neural networks",
    "epistemic aleatoric uncertainty",
    "confidence calibration deep learning",
    "uncertainty estimation prediction"
]

YEAR_START, YEAR_END = 2020, 2025
SINCE = "2020-01-01"

# Reduced target quantities to improve quality
TARGET_OPENALEX = 500      # Reduced from 1000 to 500
TARGET_OPENREVIEW = 300    # Reduced from 1200 to 300
TARGET_UNPAYWALL = 500     # Reduced from 1200 to 500

# API Configuration
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "xiang.siq@northeastern.edu").strip()
OPENREVIEW_MAILTO = os.environ.get("OPENREVIEW_MAILTO", "xiang.siq@northeastern.edu").strip()

OPENALEX_BASE = "https://api.openalex.org"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
OPENREVIEW_BASE = "https://api.openreview.net"

DEFAULT_USER_AGENT = "AAI6610-Pipeline/1.0 (mailto:xiang.siq@northeastern.edu)"

# ========== Utility Functions ==========
def ensure_dirs():
    """Create output directories"""
    OUT_OPENALEX.mkdir(parents=True, exist_ok=True)
    OUT_OPENREVIEW.mkdir(parents=True, exist_ok=True)
    OUT_UNPAYWALL.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str, limit: int = 150) -> str:
    """Clean filename"""
    name = re.sub(r"[\\/*?\"<>|:]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:limit]

def make_session(user_agent: str) -> requests.Session:
    """Create Session"""
    s = requests.Session()
    r = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": user_agent})
    return s

def backoff_sleep(i: int):
    """Exponential backoff"""
    time.sleep(min(2 ** i, 10))

def get_json(url: str, params: dict = None, headers: dict = None,
             timeout: int = 30, tries: int = 5) -> Optional[dict]:
    """JSON request with retry"""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 502, 503, 504):
                print(f"      Rate limited, waiting...")
                backoff_sleep(i + 2)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == tries - 1:
                print(f"      Request failed: {e}")
                return None
            backoff_sleep(i + 1)
    return None

def download_pdf(url: str, dest: Path, session: requests.Session, tries: int = 3) -> bool:
    """Download PDF - Optimized version"""
    for i in range(tries):
        try:
            with session.get(url, stream=True, timeout=120, allow_redirects=True) as r:
                if r.status_code >= 400:
                    if r.status_code in (429, 502, 503, 504):
                        backoff_sleep(i + 1)
                        continue
                    return False
                
                ctype = r.headers.get("Content-Type", "").lower()
                if "pdf" not in ctype and not url.lower().endswith(".pdf"):
                    return False
                
                # Check file size
                total = int(r.headers.get("Content-Length", 0))
                if total > 50 * 1024 * 1024:  # Skip files >50MB
                    return False
                
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(65536):
                        if chunk:
                            f.write(chunk)
                
                return True
        except Exception:
            if i == tries - 1:
                return False
            backoff_sleep(i + 1)
    return False

def write_metadata_txt(txt_path: Path, title: str, abstract: str, date: str, doi: str, source: str):
    """Save metadata"""
    txt = [
        f"Title: {title or 'N/A'}",
        f"Date: {date or 'N/A'}",
        f"DOI: {doi or 'N/A'}",
        f"Source: {source}",
        "",
        "Abstract:",
        textwrap.fill(abstract or "N/A", width=100),
        "",
        "="*80,
        "",
        "Full Text: (PDF conversion will be done by main pipeline)"
    ]
    
    txt_path.write_text("\n".join(txt), encoding="utf-8")

def norm_title(s: str) -> str:
    """Normalize title"""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

# ========== OpenAlex ==========
def reconstruct_abstract(inv_idx: Optional[Dict[str, List[int]]]) -> str:
    """Reconstruct abstract"""
    if not inv_idx:
        return ""
    pos = []
    for token, idxs in inv_idx.items():
        for i in idxs:
            pos.append((i, token))
    pos.sort(key=lambda x: x[0])
    return " ".join(tok for _, tok in pos)

def pick_pdf_from_openalex(work: Dict[str, Any]) -> Optional[str]:
    """Extract PDF URL - Optimized version, try multiple sources"""
    # Priority 1: best_oa_location
    b = work.get("best_oa_location")
    if b and b.get("pdf_url"):
        return b["pdf_url"]
    
    # Priority 2: All oa_locations
    for loc in work.get("oa_locations") or []:
        if loc.get("pdf_url"):
            return loc["pdf_url"]
    
    # Priority 3: primary_location
    p = work.get("primary_location") or {}
    if p.get("pdf_url"):
        return p["pdf_url"]
    
    # Priority 4: Construct possible PDF link from landing_page_url
    landing = p.get("landing_page_url") or ""
    if "arxiv.org/abs/" in landing:
        arxiv_id = landing.split("/abs/")[-1]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    return None

def openalex_fetch(session: requests.Session, query: str, since: str, max_results: int) -> List[Dict[str, Any]]:
    """Fetch papers from OpenAlex"""
    base = {"search": query, "sort": "publication_date:desc"}
    if since:
        base["filter"] = f"from_publication_date:{since}"
    
    works: List[Dict[str, Any]] = []
    cursor = "*"
    per_page = 200
    
    print(f"  Fetching from OpenAlex...")
    
    while len(works) < max_results and cursor:
        q = {**base, "per_page": per_page, "cursor": cursor}
        d = get_json(f"{OPENALEX_BASE}/works", params=q, timeout=60, tries=5)
        
        if not d:
            print(f"      Request failed, stopping")
            break
        
        batch = d.get("results", [])
        works.extend(batch)
        cursor = d.get("meta", {}).get("next_cursor")
        
        print(f"      Fetched: {len(works)}/{max_results}")
        
        time.sleep(0.5)  # Increased delay to avoid rate limiting
        
        if not batch:
            break
    
    print(f"  Total from OpenAlex: {len(works[:max_results])}")
    return works[:max_results]

def pipeline_openalex(session: requests.Session, target: int) -> List[dict]:
    """OpenAlex pipeline - Merge multiple queries"""
    all_works = []
    seen_titles: Set[str] = set()
    
    # Use multiple query terms to increase coverage
    for query in QUERIES[:3]:  # Only use first 3 queries to avoid duplication
        print(f"\n  Query: {query}")
        works = openalex_fetch(session, query, SINCE, target)
        
        # Deduplicate
        for w in works:
            title = w.get("title") or ""
            tkey = norm_title(title)
            if tkey and tkey not in seen_titles:
                seen_titles.add(tkey)
                all_works.append(w)
        
        print(f"      Unique works so far: {len(all_works)}")
        
        if len(all_works) >= target * 2:
            break
    
    print(f"\n  Total unique works: {len(all_works)}")
    print(f"  Downloading PDFs (target: {target})...")
    
    downloaded = []
    failed = 0
    
    for idx, w in enumerate(all_works):
        if len(downloaded) >= target:
            break
        
        title = w.get("title") or "untitled"
        date = w.get("publication_date") or ""
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        
        pdf_url = pick_pdf_from_openalex(w)
        if not pdf_url:
            failed += 1
            continue
        
        fname = sanitize_filename(f"{title[:80]}_{doi or idx}")
        pdf_path = OUT_OPENALEX / f"{fname}.pdf"
        
        # Skip existing files
        if pdf_path.exists():
            txt_path = pdf_path.with_suffix(".txt")
            downloaded.append({
                "title": title,
                "abstract": abstract,
                "date": date,
                "doi": doi,
                "pdf_path": str(pdf_path),
                "txt_path": str(txt_path)
            })
            continue
        
        if download_pdf(pdf_url, pdf_path, session):
            txt_path = pdf_path.with_suffix(".txt")
            write_metadata_txt(txt_path, title, abstract, date, doi, "OpenAlex")
            
            downloaded.append({
                "title": title,
                "abstract": abstract,
                "date": date,
                "doi": doi,
                "pdf_path": str(pdf_path),
                "txt_path": str(txt_path)
            })
            
            if len(downloaded) % 50 == 0:
                print(f"      Progress: {len(downloaded)}/{target} (failed: {failed})")
        else:
            failed += 1
        
        time.sleep(0.5)  # Avoid rate limiting
    
    print(f"  Downloaded: {len(downloaded)} PDFs (failed: {failed})")
    return downloaded

# ========== OpenReview ==========
def openreview_search(query: str, limit: int) -> List[dict]:
    """Search OpenReview - Optimized version, use keywords instead of full query"""
    url = f"{OPENREVIEW_BASE}/notes"
    out: List[dict] = []
    per_page = 100
    offset = 0
    
    print(f"  Searching OpenReview...")
    
    # Extract keywords
    keywords = ["uncertainty", "prediction", "machine learning", "deep learning"]
    
    for keyword in keywords:
        print(f"      Trying keyword: {keyword}")
        
        params = {
            "content.keywords": keyword,
            "limit": per_page,
            "offset": 0
        }
        
        if OPENREVIEW_MAILTO:
            params["mailto"] = OPENREVIEW_MAILTO
        
        data = get_json(url, params=params, timeout=30, tries=4)
        
        if not data or not data.get("notes"):
            continue
        
        for n in data["notes"]:
            if len(out) >= limit:
                break
            
            c = n.get("content", {})
            
            # Extract fields
            title = c.get("title", "")
            if isinstance(title, dict):
                title = title.get("value", "")
            
            abstract = c.get("abstract", "")
            if isinstance(abstract, dict):
                abstract = abstract.get("value", "")
            
            year = c.get("year", "")
            if isinstance(year, dict):
                year = year.get("value", "")
            
            if not title or not str(year).isdigit():
                continue
            
            y = int(year)
            if not (YEAR_START <= y <= YEAR_END):
                continue
            
            note_id = n.get("id")
            if not note_id:
                continue
            
            out.append({
                "title": title,
                "abstract": abstract or "",
                "date": str(y),
                "doi": "",
                "pdf_url": f"https://openreview.net/pdf?id={note_id}"
            })
        
        time.sleep(1)
    
    print(f"  Found {len(out)} papers from OpenReview")
    return out[:limit]

def pipeline_openreview(session: requests.Session, target: int) -> List[dict]:
    """OpenReview pipeline"""
    items = openreview_search("uncertainty", target)
    
    if not items:
        print("  No papers found, trying alternative API...")
        return []
    
    downloaded = []
    seen = set()
    failed = 0
    
    print(f"  Downloading PDFs (target: {target})...")
    
    for r in items:
        if len(downloaded) >= target:
            break
        
        tkey = norm_title(r["title"])
        if tkey in seen:
            continue
        seen.add(tkey)
        
        pdf_url = r.get("pdf_url")
        if not pdf_url:
            failed += 1
            continue
        
        pdf_path = OUT_OPENREVIEW / (sanitize_filename(r["title"]) + ".pdf")
        
        if pdf_path.exists():
            downloaded.append({**r, "pdf_path": str(pdf_path)})
            continue
        
        if download_pdf(pdf_url, pdf_path, session):
            txt_path = pdf_path.with_suffix(".txt")
            write_metadata_txt(
                txt_path,
                r["title"],
                r.get("abstract", ""),
                r.get("date", ""),
                r.get("doi", ""),
                "OpenReview"
            )
            
            downloaded.append({
                **r,
                "pdf_path": str(pdf_path),
                "txt_path": str(txt_path)
            })
            
            if len(downloaded) % 50 == 0:
                print(f"      Progress: {len(downloaded)}/{target} (failed: {failed})")
        else:
            failed += 1
        
        time.sleep(1)
    
    print(f"  Downloaded: {len(downloaded)} PDFs (failed: {failed})")
    return downloaded

# ========== Unpaywall ==========
def openalex_get_dois(session: requests.Session, max_results: int, exclude_titles: set, exclude_dois: set) -> List[dict]:
    """Get DOIs from OpenAlex"""
    candidates = []
    seen = set()
    
    for query in QUERIES[:2]:
        print(f"\n  Fetching DOIs for: {query}")
        works = openalex_fetch(session, query, SINCE, max_results)
        
        for w in works:
            if len(candidates) >= max_results:
                break
            
            title = w.get("title") or ""
            tkey = norm_title(title)
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            doi_low = doi.lower()
            
            if not title or not doi or tkey in exclude_titles or doi_low in exclude_dois:
                continue
            
            if doi_low in seen:
                continue
            seen.add(doi_low)
            
            date = w.get("publication_date") or ""
            abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
            
            candidates.append({
                "title": title,
                "abstract": abstract,
                "date": date,
                "doi": doi
            })
    
    return candidates

def unpaywall_get_pdf_url(session: requests.Session, doi: str, email: str) -> Optional[str]:
    """Get PDF URL via Unpaywall"""
    if not doi or not email:
        return None
    
    url = f"{UNPAYWALL_BASE}/{quote_plus(doi)}"
    data = get_json(url, params={"email": email}, timeout=60, tries=4)
    
    if not data:
        return None
    
    loc = data.get("best_oa_location") or {}
    if loc.get("url_for_pdf"):
        return loc["url_for_pdf"]
    
    for loc in data.get("oa_locations") or []:
        if loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
    
    return None

def pipeline_unpaywall(session: requests.Session, target: int,
                      exclude_titles: set, exclude_dois: set) -> List[dict]:
    """Unpaywall pipeline"""
    if not UNPAYWALL_EMAIL:
        print("  UNPAYWALL_EMAIL not set, skipping")
        return []
    
    candidates = openalex_get_dois(session, target * 3, exclude_titles, exclude_dois)
    print(f"  Found {len(candidates)} DOI candidates")
    
    downloaded = []
    failed = 0
    
    print(f"  Downloading via Unpaywall (target: {target})...")
    
    for r in candidates:
        if len(downloaded) >= target:
            break
        
        doi = r.get("doi", "").strip()
        pdf_url = unpaywall_get_pdf_url(session, doi, UNPAYWALL_EMAIL)
        
        if not pdf_url:
            failed += 1
            continue
        
        pdf_path = OUT_UNPAYWALL / (sanitize_filename(r["title"]) + ".pdf")
        
        if pdf_path.exists():
            downloaded.append({**r, "pdf_path": str(pdf_path)})
            continue
        
        if download_pdf(pdf_url, pdf_path, session):
            txt_path = pdf_path.with_suffix(".txt")
            write_metadata_txt(
                txt_path,
                r["title"],
                r.get("abstract", ""),
                r.get("date", ""),
                doi,
                "Unpaywall"
            )
            
            downloaded.append({
                **r,
                "pdf_path": str(pdf_path),
                "txt_path": str(txt_path)
            })
            
            if len(downloaded) % 50 == 0:
                print(f"      Progress: {len(downloaded)}/{target} (failed: {failed})")
        else:
            failed += 1
        
        time.sleep(2)  # Unpaywall has rate limits
    
    print(f"  Downloaded: {len(downloaded)} PDFs (failed: {failed})")
    return downloaded

# ========== Main Program ==========
def main():
    """Main program"""
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print("OpenAlex + OpenReview + Unpaywall Crawler (OPTIMIZED)")
    print(f"{'='*70}")
    print(f"Queries: {len(QUERIES)} search terms")
    print(f"Years: {YEAR_START}-{YEAR_END}")
    print(f"Targets: OpenAlex={TARGET_OPENALEX}, OpenReview={TARGET_OPENREVIEW}, Unpaywall={TARGET_UNPAYWALL}")
    print(f"Output: {BASE_OUT}")
    print(f"{'='*70}\n")
    
    ensure_dirs()
    session = make_session(DEFAULT_USER_AGENT)
    
    all_downloads = []
    
    # [1/3] OpenAlex
    print(f"{'='*70}")
    print("[1/3] OpenAlex")
    print(f"{'='*70}")
    dl_openalex = pipeline_openalex(session, TARGET_OPENALEX)
    print(f"OpenAlex: {len(dl_openalex)} PDFs\n")
    all_downloads.extend([{**d, "source": "OpenAlex"} for d in dl_openalex])
    
    # [2/3] OpenReview
    print(f"{'='*70}")
    print("[2/3] OpenReview")
    print(f"{'='*70}")
    dl_openreview = pipeline_openreview(session, TARGET_OPENREVIEW)
    print(f"OpenReview: {len(dl_openreview)} PDFs\n")
    all_downloads.extend([{**d, "source": "OpenReview"} for d in dl_openreview])
    
    # Deduplicate
    ex_titles = {norm_title(x["title"]) for x in all_downloads}
    ex_dois = {(x.get("doi") or "").lower() for x in all_downloads if x.get("doi")}
    
    # [3/3] Unpaywall
    print(f"{'='*70}")
    print("[3/3] Unpaywall")
    print(f"{'='*70}")
    dl_unpaywall = pipeline_unpaywall(session, TARGET_UNPAYWALL, ex_titles, ex_dois)
    print(f"Unpaywall: {len(dl_unpaywall)} PDFs\n")
    all_downloads.extend([{**d, "source": "Unpaywall"} for d in dl_unpaywall])
    
    # Generate report
    report_path = BASE_OUT / "download_summary.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "date", "doi", "source", "pdf_path", "txt_path"])
        writer.writeheader()
        
        for r in all_downloads:
            writer.writerow({
                "title": r.get("title", ""),
                "date": r.get("date", ""),
                "doi": r.get("doi", "") or "",
                "source": r.get("source", ""),
                "pdf_path": r.get("pdf_path", ""),
                "txt_path": r.get("txt_path", "")
            })
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"OpenAlex    : {len(dl_openalex):>4} PDFs")
    print(f"OpenReview  : {len(dl_openreview):>4} PDFs")
    print(f"Unpaywall   : {len(dl_unpaywall):>4} PDFs")
    print(f"{'─'*70}")
    print(f"TOTAL       : {len(all_downloads):>4} documents")
    print(f"Time        : {elapsed/60:.1f} minutes")
    print(f"Success Rate: {len(all_downloads)/sum([TARGET_OPENALEX, TARGET_OPENREVIEW, TARGET_UNPAYWALL])*100:.1f}%")
    print(f"{'='*70}")
    print(f"Output      : {BASE_OUT}")
    print(f"Report      : {report_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
