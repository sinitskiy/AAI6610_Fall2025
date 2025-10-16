#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import time
import json
import argparse
from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OPENALEX_BASE = "https://api.openalex.org"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
DEFAULT_USER_AGENT = "Script-E-PaperCrawler/1.5 (mailto:jerichowang@outlook.com)"

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sanitize_filename(name: str, max_len: int = 150) -> str:
    name = re.sub(r"[^\w\-\.,\(\) \[\]&]+", "_", name)
    return (name[:max_len]).strip(" ._")

def make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    r = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": user_agent})
    return s

def safe_get_json(session: requests.Session, url: str, params: dict = None, timeout: int = 60) -> Optional[dict]:
    try:
        resp = session.get(url, params=params, timeout=timeout)
        if resp.status_code == 400:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

def reconstruct_abstract(inv_idx: Optional[Dict[str, List[int]]]) -> str:
    if not inv_idx:
        return ""
    pos = []
    for token, idxs in inv_idx.items():
        for i in idxs:
            pos.append((i, token))
    pos.sort(key=lambda x: x[0])
    return " ".join(tok for _, tok in pos)

def pick_pdf(work: Dict[str, Any]) -> Optional[str]:
    b = work.get("best_oa_location")
    if b and b.get("pdf_url"):
        return b["pdf_url"]
    for loc in work.get("oa_locations") or []:
        if loc.get("pdf_url"):
            return loc["pdf_url"]
    p = work.get("primary_location") or {}
    if p.get("pdf_url"):
        return p["pdf_url"]
    return None

def download_pdf(session: requests.Session, url: str, out_path: str) -> bool:
    try:
        with session.get(url, timeout=90, stream=True, allow_redirects=True) as r:
            if r.status_code >= 400:
                return False
            ctype = r.headers.get("Content-Type", "").lower()
            if ("pdf" not in ctype) and (not url.lower().endswith(".pdf")):
                return False
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False

def unpaywall_pdf(session: requests.Session, doi: str, email: Optional[str]) -> Optional[str]:
    if not doi or not email or "@" not in email:
        return None
    data = safe_get_json(session, f"{UNPAYWALL_BASE}/{quote_plus(doi)}", {"email": email}, timeout=60)
    if not data:
        return None
    b = data.get("best_oa_location") or {}
    if b.get("url_for_pdf"):
        return b["url_for_pdf"]
    for loc in data.get("oa_locations") or []:
        if loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
    return None

def openalex_fetch(session: requests.Session, query: str, since: str, max_results: int) -> List[Dict[str, Any]]:
    base = {"search": query, "sort": "publication_date:desc"}
    if since:
        base["filter"] = f"from_publication_date:{since}"
    works: List[Dict[str, Any]] = []
    cursor = "*"
    per_page = 200
    while len(works) < max_results and cursor:
        q = {**base, "per_page": per_page, "cursor": cursor}
        d = safe_get_json(session, f"{OPENALEX_BASE}/works", q, timeout=60)
        if not d:
            break
        batch = d.get("results", [])
        works.extend(batch)
        cursor = d.get("meta", {}).get("next_cursor")
        time.sleep(0.25)
        if not batch:
            break
    return works[:max_results]

def write_metadata_txt(path: str, meta: Dict[str, Any]) -> None:
    txt = [
        f"Title: {meta.get('title','')}",
        f"Date: {meta.get('date','')}",
        f"DOI: {meta.get('doi','')}",
        f"URL: {meta.get('url','')}",
        "Abstract:",
        meta.get("abstract", "") or "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))

def process_items(session: requests.Session, works: List[Dict[str, Any]], out_dir: str, target_pdfs: int, unpaywall_email: Optional[str]) -> Dict[str, int]:
    pdf_dir = os.path.join(out_dir, "pdfs")
    meta_dir = os.path.join(out_dir, "metadata")
    ensure_dir(pdf_dir)
    ensure_dir(meta_dir)
    meta_ct = 0
    pdf_ct = 0
    for w in works:
        if pdf_ct >= target_pdfs:
            break
        title = w.get("title") or "untitled"
        date = w.get("publication_date") or ""
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        url = w.get("primary_location", {}).get("landing_page_url") or w.get("id")
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        pdf_url = pick_pdf(w)
        fname = sanitize_filename(f"{title[:80]}_{doi or w.get('id','')}")
        pdf_path = os.path.join(pdf_dir, f"{fname}.pdf")
        txt_path = os.path.join(meta_dir, f"{fname}.txt")
        downloaded = False
        if pdf_url and download_pdf(session, pdf_url, pdf_path):
            downloaded = True
        if not downloaded and doi:
            up = unpaywall_pdf(session, doi, unpaywall_email)
            if up and download_pdf(session, up, pdf_path):
                downloaded = True
        write_metadata_txt(txt_path, {
            "title": title, "date": date, "doi": doi, "url": url, "abstract": abstract
        })
        meta_ct += 1
        if downloaded:
            pdf_ct += 1
        if meta_ct % 50 == 0:
            print(f"Processed {meta_ct} items | PDFs saved: {pdf_ct}")
    return {"metadata": meta_ct, "pdfs": pdf_ct}

def main():
    ap = argparse.ArgumentParser(description="Script E — force-download PDFs via OpenAlex + Unpaywall fallback.")
    ap.add_argument("--query", default="uncertainty prediction ML")
    ap.add_argument("--since", default="2011-01-01")
    ap.add_argument("--max-fetch", type=int, default=5000)
    ap.add_argument("--target-pdfs", type=int, default=1000)
    ap.add_argument("--out", default="./")
    ap.add_argument("--unpaywall-email", default="wang.jialin2@northeastern.edu")
    ap.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    args = ap.parse_args()
    ensure_dir(args.out)
    session = make_session(args.user_agent)
    print(f"Fetching up to {args.max_fetch} metadata...")
    works = openalex_fetch(session, args.query, args.since, args.max_fetch)
    print(f"Got {len(works)} metadata records. Target PDFs: {args.target_pdfs}")
    stats = process_items(session, works, args.out, args.target_pdfs, args.unpaywall_email)
    report = {
        "query": args.query,
        "since": args.since,
        "max_fetch": args.max_fetch,
        "target_pdfs": args.target_pdfs,
        "out_dir": args.out,
        "result": stats,
    }
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(report)

if __name__ == "__main__":
    main()
