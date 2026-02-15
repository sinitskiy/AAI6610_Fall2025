#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenAlex + OpenReview + Unpaywall Scraper v2.0
- Fixed OpenReview API (uses correct endpoint)
- Reads settings from config.yaml
- Improved error handling and rate limiting
- Better deduplication
"""

import os
import re
import sys
import time
import json
import csv
import logging
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from urllib.parse import quote_plus
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

# ============================================================================
# Setup Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Path Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
SCRAPERS_DIR = SCRIPT_DIR.parent
PIPELINE_DIR = SCRAPERS_DIR.parent
PROJECT_ROOT = PIPELINE_DIR.parent

CONFIG_PATH = PIPELINE_DIR / "config.yaml"
ENV_PATH = PIPELINE_DIR / ".env"

# Output directories
BASE_OUT = SCRAPERS_DIR / "outputs" / "openalex_openreview_papers"
OUT_OPENALEX = BASE_OUT / "openalex"
OUT_OPENREVIEW = BASE_OUT / "openreview"
OUT_UNPAYWALL = BASE_OUT / "unpaywall"

# ============================================================================
# Load Environment & Config
# ============================================================================
def load_env():
    """Load .env file"""
    for env_path in [ENV_PATH, PROJECT_ROOT / ".env"]:
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip())
            logger.info(f"Loaded env from {env_path.name}")
            return

load_env()

def load_config() -> dict:
    """Load configuration"""
    for path in [CONFIG_PATH, PROJECT_ROOT / "config.yaml"]:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except:
                pass
    return {}

# ============================================================================
# Configuration
# ============================================================================
CONFIG = load_config()
SCRAPER_CONFIG = CONFIG.get('scrapers', {}).get('openalex', {})

# API settings
UNPAYWALL_EMAIL = os.environ.get('UNPAYWALL_EMAIL', '').strip()
OPENREVIEW_MAILTO = os.environ.get('OPENREVIEW_MAILTO', '').strip()
OPENALEX_EMAIL = os.environ.get('OPENALEX_USER_AGENT', '').strip()

# Targets (increased for more data)
TARGET_OPENALEX = SCRAPER_CONFIG.get('target_openalex', 800)
TARGET_OPENREVIEW = SCRAPER_CONFIG.get('target_openreview', 500)
TARGET_UNPAYWALL = SCRAPER_CONFIG.get('target_unpaywall', 600)

# Date range
YEAR_START = 2020
YEAR_END = datetime.now().year

# Search queries - expanded for better coverage
QUERIES = [
    "uncertainty quantification machine learning",
    "bayesian neural networks",
    "epistemic aleatoric uncertainty",
    "confidence calibration deep learning",
    "uncertainty estimation prediction",
    "conformal prediction",
    "monte carlo dropout uncertainty",
    "deep ensemble uncertainty",
    "out-of-distribution detection",
    "predictive uncertainty neural networks",
]

# API URLs
OPENALEX_BASE = "https://api.openalex.org"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
OPENREVIEW_BASE = "https://api2.openreview.net"  # Note: api2 for newer endpoint

# ============================================================================
# Utility Functions
# ============================================================================
def ensure_dirs():
    """Create output directories"""
    for d in [OUT_OPENALEX, OUT_OPENREVIEW, OUT_UNPAYWALL]:
        d.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str, limit: int = 120) -> str:
    """Clean filename"""
    name = re.sub(r'[\\/*?\"<>|:\n\r]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'_+', '_', name)
    return name[:limit]

def norm_title(s: str) -> str:
    """Normalize title for deduplication"""
    return re.sub(r'[^a-z0-9]+', '', (s or '').lower())

def make_session(user_agent: str = None) -> requests.Session:
    """Create session with retries"""
    s = requests.Session()
    
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['HEAD', 'GET', 'OPTIONS'],
    )
    s.mount('https://', HTTPAdapter(max_retries=retry))
    
    ua = user_agent or f'AAI6610-Pipeline/2.0 (mailto:{UNPAYWALL_EMAIL})'
    s.headers.update({'User-Agent': ua})
    
    return s

def get_json(url: str, params: dict = None, session: requests.Session = None,
             timeout: int = 30, retries: int = 3) -> Optional[dict]:
    """GET JSON with retries"""
    sess = session or requests.Session()
    
    for i in range(retries):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            
            if r.status_code == 429:
                wait = min(2 ** (i + 2), 30)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            
            r.raise_for_status()
            return r.json()
            
        except Exception as e:
            if i == retries - 1:
                logger.debug(f"Request failed: {e}")
            time.sleep(2 ** i)
    
    return None

def download_pdf(url: str, dest: Path, session: requests.Session, 
                timeout: int = 120) -> bool:
    """Download PDF file"""
    try:
        with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
            if r.status_code >= 400:
                return False
            
            ctype = r.headers.get('Content-Type', '').lower()
            if 'pdf' not in ctype and not url.lower().endswith('.pdf'):
                if 'html' in ctype:  # Likely a paywall page
                    return False
            
            size = int(r.headers.get('Content-Length', 0))
            if size > 50 * 1024 * 1024:  # Skip >50MB
                return False
            
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(65536):
                    if chunk:
                        f.write(chunk)
            
            # Verify it's actually a PDF
            if dest.exists() and dest.stat().st_size > 1000:
                with open(dest, 'rb') as f:
                    header = f.read(5)
                    if header != b'%PDF-':
                        dest.unlink()
                        return False
            
            return True
            
    except Exception as e:
        logger.debug(f"Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False

def write_metadata_txt(path: Path, title: str, abstract: str, 
                       date: str, doi: str, source: str, authors: str = ""):
    """Save metadata as TXT"""
    lines = [
        f"Title: {title or 'N/A'}",
        f"Authors: {authors or 'N/A'}",
        f"Date: {date or 'N/A'}",
        f"DOI: {doi or 'N/A'}",
        f"Source: {source}",
        "",
        "Abstract:",
        textwrap.fill(abstract or 'N/A', width=100),
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')

# ============================================================================
# OpenAlex Scraper
# ============================================================================
class OpenAlexScraper:
    """Scraper for OpenAlex API"""
    
    def __init__(self, session: requests.Session):
        self.session = session
        self.seen_titles: Set[str] = set()
    
    def reconstruct_abstract(self, inv_idx: Optional[Dict[str, List[int]]]) -> str:
        """Reconstruct abstract from inverted index"""
        if not inv_idx:
            return ""
        
        pos = []
        for token, idxs in inv_idx.items():
            for i in idxs:
                pos.append((i, token))
        pos.sort()
        return ' '.join(tok for _, tok in pos)
    
    def get_pdf_url(self, work: dict) -> Optional[str]:
        """Extract best PDF URL"""
        # Try best_oa_location
        best = work.get('best_oa_location') or {}
        if best.get('pdf_url'):
            return best['pdf_url']
        
        # Try all oa_locations
        for loc in work.get('oa_locations') or []:
            if loc.get('pdf_url'):
                return loc['pdf_url']
        
        # Try primary_location
        primary = work.get('primary_location') or {}
        if primary.get('pdf_url'):
            return primary['pdf_url']
        
        # Try to construct arXiv URL
        landing = primary.get('landing_page_url') or ''
        if 'arxiv.org/abs/' in landing:
            arxiv_id = landing.split('/abs/')[-1]
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        return None
    
    def fetch_works(self, query: str, max_results: int = 300) -> List[dict]:
        """Fetch works from OpenAlex"""
        works = []
        cursor = '*'
        per_page = 200
        
        params_base = {
            'search': query,
            'filter': f'from_publication_date:{YEAR_START}-01-01',
            'sort': 'publication_date:desc',
            'per_page': per_page,
        }
        
        if OPENALEX_EMAIL:
            params_base['mailto'] = OPENALEX_EMAIL
        
        while len(works) < max_results and cursor:
            params = {**params_base, 'cursor': cursor}
            
            data = get_json(f"{OPENALEX_BASE}/works", params=params, 
                           session=self.session, timeout=60)
            
            if not data:
                break
            
            batch = data.get('results', [])
            works.extend(batch)
            
            cursor = data.get('meta', {}).get('next_cursor')
            
            if not batch:
                break
            
            time.sleep(0.3)
        
        return works[:max_results]
    
    def scrape(self, target: int) -> List[dict]:
        """Main scrape function"""
        logger.info("Fetching from OpenAlex...")
        
        all_works = []
        
        for query in QUERIES[:5]:  # Use top 5 queries
            logger.info(f"  Query: {query[:50]}...")
            
            works = self.fetch_works(query, max_results=target)
            
            for w in works:
                title = w.get('title') or ''
                tkey = norm_title(title)
                
                if tkey and tkey not in self.seen_titles:
                    self.seen_titles.add(tkey)
                    all_works.append(w)
            
            logger.info(f"    Unique: {len(all_works)}")
            
            if len(all_works) >= target * 2:
                break
        
        logger.info(f"Total unique works: {len(all_works)}")
        
        # Download PDFs
        downloaded = []
        failed = 0
        
        for idx, w in enumerate(all_works):
            if len(downloaded) >= target:
                break
            
            title = w.get('title') or 'untitled'
            date = w.get('publication_date') or ''
            doi = (w.get('doi') or '').replace('https://doi.org/', '')
            abstract = self.reconstruct_abstract(w.get('abstract_inverted_index'))
            
            # Get authors
            authors = []
            for auth in (w.get('authorships') or [])[:5]:
                name = auth.get('author', {}).get('display_name')
                if name:
                    authors.append(name)
            
            pdf_url = self.get_pdf_url(w)
            if not pdf_url:
                failed += 1
                continue
            
            fname = sanitize_filename(f"{title[:80]}_{doi or idx}")
            pdf_path = OUT_OPENALEX / f"{fname}.pdf"
            txt_path = OUT_OPENALEX / f"{fname}.txt"
            
            # Skip if exists
            if txt_path.exists():
                downloaded.append({'title': title, 'doi': doi})
                continue
            
            if download_pdf(pdf_url, pdf_path, self.session):
                write_metadata_txt(txt_path, title, abstract, date, doi, 
                                 "OpenAlex", ', '.join(authors))
                downloaded.append({
                    'title': title,
                    'abstract': abstract,
                    'date': date,
                    'doi': doi,
                    'authors': authors,
                })
                
                if len(downloaded) % 50 == 0:
                    logger.info(f"    Progress: {len(downloaded)}/{target}")
            else:
                failed += 1
            
            time.sleep(0.5)
        
        logger.info(f"OpenAlex: {len(downloaded)} downloaded, {failed} failed")
        return downloaded

# ============================================================================
# OpenReview Scraper (FIXED)
# ============================================================================
class OpenReviewScraper:
    """Scraper for OpenReview API - Fixed version"""
    
    # Major ML venues on OpenReview
    VENUES = [
        'ICLR.cc/2024',
        'ICLR.cc/2023',
        'NeurIPS.cc/2024',
        'NeurIPS.cc/2023',
        'ICML.cc/2024',
        'ICML.cc/2023',
    ]
    
    KEYWORDS = ['uncertainty', 'bayesian', 'calibration', 'conformal', 
                'probabilistic', 'ensemble']
    
    def __init__(self, session: requests.Session):
        self.session = session
        self.seen_titles: Set[str] = set()
    
    def search_venue(self, venue: str, limit: int = 100) -> List[dict]:
        """Search papers in a specific venue"""
        results = []
        
        # Use the notes/search endpoint
        url = f"{OPENREVIEW_BASE}/notes/search"
        
        params = {
            'query': 'uncertainty OR bayesian OR calibration',
            'group': venue,
            'limit': min(limit, 100),
            'offset': 0,
        }
        
        try:
            data = get_json(url, params=params, session=self.session, timeout=30)
            
            if not data:
                # Try alternative endpoint
                url = f"{OPENREVIEW_BASE}/notes"
                params = {
                    'invitation': f'{venue}/-/Submission',
                    'limit': limit,
                }
                data = get_json(url, params=params, session=self.session, timeout=30)
            
            if data and 'notes' in data:
                for note in data['notes']:
                    content = note.get('content', {})
                    
                    # Handle different content structures
                    title = content.get('title', {})
                    if isinstance(title, dict):
                        title = title.get('value', '')
                    
                    abstract = content.get('abstract', {})
                    if isinstance(abstract, dict):
                        abstract = abstract.get('value', '')
                    
                    if not title:
                        continue
                    
                    # Check if uncertainty-related
                    text = f"{title} {abstract}".lower()
                    if not any(kw in text for kw in self.KEYWORDS):
                        continue
                    
                    note_id = note.get('id')
                    if not note_id:
                        continue
                    
                    results.append({
                        'title': title,
                        'abstract': abstract or '',
                        'venue': venue,
                        'note_id': note_id,
                        'pdf_url': f"https://openreview.net/pdf?id={note_id}",
                    })
        
        except Exception as e:
            logger.debug(f"OpenReview search error: {e}")
        
        return results
    
    def scrape(self, target: int, exclude_titles: Set[str] = None) -> List[dict]:
        """Main scrape function"""
        logger.info("Fetching from OpenReview...")
        
        exclude = exclude_titles or set()
        all_papers = []
        
        for venue in self.VENUES:
            logger.info(f"  Venue: {venue}")
            
            papers = self.search_venue(venue, limit=150)
            
            for p in papers:
                tkey = norm_title(p['title'])
                if tkey and tkey not in self.seen_titles and tkey not in exclude:
                    self.seen_titles.add(tkey)
                    all_papers.append(p)
            
            logger.info(f"    Found: {len(papers)}, Unique total: {len(all_papers)}")
            time.sleep(1)
            
            if len(all_papers) >= target * 1.5:
                break
        
        logger.info(f"Total unique papers: {len(all_papers)}")
        
        # Download PDFs
        downloaded = []
        failed = 0
        
        for p in all_papers:
            if len(downloaded) >= target:
                break
            
            fname = sanitize_filename(p['title'])
            pdf_path = OUT_OPENREVIEW / f"{fname}.pdf"
            txt_path = OUT_OPENREVIEW / f"{fname}.txt"
            
            if txt_path.exists():
                downloaded.append(p)
                continue
            
            if download_pdf(p['pdf_url'], pdf_path, self.session):
                write_metadata_txt(txt_path, p['title'], p['abstract'],
                                 p.get('venue', ''), '', "OpenReview")
                downloaded.append(p)
                
                if len(downloaded) % 50 == 0:
                    logger.info(f"    Progress: {len(downloaded)}/{target}")
            else:
                failed += 1
            
            time.sleep(1)
        
        logger.info(f"OpenReview: {len(downloaded)} downloaded, {failed} failed")
        return downloaded

# ============================================================================
# Unpaywall Scraper
# ============================================================================
class UnpaywallScraper:
    """Scraper using Unpaywall API"""
    
    def __init__(self, session: requests.Session, email: str):
        self.session = session
        self.email = email
    
    def get_pdf_url(self, doi: str) -> Optional[str]:
        """Get PDF URL from Unpaywall"""
        if not doi or not self.email:
            return None
        
        url = f"{UNPAYWALL_BASE}/{quote_plus(doi)}"
        data = get_json(url, params={'email': self.email}, 
                       session=self.session, timeout=60)
        
        if not data:
            return None
        
        # Try best OA location first
        best = data.get('best_oa_location') or {}
        if best.get('url_for_pdf'):
            return best['url_for_pdf']
        
        # Try all locations
        for loc in data.get('oa_locations') or []:
            if loc.get('url_for_pdf'):
                return loc['url_for_pdf']
        
        return None
    
    def scrape(self, candidates: List[dict], target: int) -> List[dict]:
        """Download papers via Unpaywall"""
        if not self.email:
            logger.warning("UNPAYWALL_EMAIL not set, skipping")
            return []
        
        logger.info(f"Processing {len(candidates)} DOI candidates via Unpaywall...")
        
        downloaded = []
        failed = 0
        
        for r in candidates:
            if len(downloaded) >= target:
                break
            
            doi = r.get('doi', '').strip()
            if not doi:
                continue
            
            pdf_url = self.get_pdf_url(doi)
            if not pdf_url:
                failed += 1
                continue
            
            fname = sanitize_filename(r.get('title', doi)[:80])
            pdf_path = OUT_UNPAYWALL / f"{fname}.pdf"
            txt_path = OUT_UNPAYWALL / f"{fname}.txt"
            
            if txt_path.exists():
                downloaded.append(r)
                continue
            
            if download_pdf(pdf_url, pdf_path, self.session):
                write_metadata_txt(txt_path, r.get('title', ''), 
                                 r.get('abstract', ''), r.get('date', ''),
                                 doi, "Unpaywall")
                downloaded.append(r)
                
                if len(downloaded) % 50 == 0:
                    logger.info(f"    Progress: {len(downloaded)}/{target}")
            else:
                failed += 1
            
            time.sleep(2)  # Unpaywall rate limit
        
        logger.info(f"Unpaywall: {len(downloaded)} downloaded, {failed} failed")
        return downloaded

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    start_time = time.time()
    
    print(f"\n{'#'*70}")
    print("# OpenAlex + OpenReview + Unpaywall Scraper v2.0")
    print(f"{'#'*70}")
    print(f"Targets: OpenAlex={TARGET_OPENALEX}, OpenReview={TARGET_OPENREVIEW}, Unpaywall={TARGET_UNPAYWALL}")
    print(f"Years: {YEAR_START}-{YEAR_END}")
    print(f"Output: {BASE_OUT}")
    print(f"{'#'*70}\n")
    
    ensure_dirs()
    session = make_session()
    
    all_downloads = []
    
    # 1. OpenAlex
    print(f"\n{'='*60}")
    print("[1/3] OpenAlex")
    print(f"{'='*60}")
    
    openalex = OpenAlexScraper(session)
    dl_openalex = openalex.scrape(TARGET_OPENALEX)
    all_downloads.extend([{**d, 'source': 'OpenAlex'} for d in dl_openalex])
    
    # 2. OpenReview
    print(f"\n{'='*60}")
    print("[2/3] OpenReview")
    print(f"{'='*60}")
    
    exclude_titles = {norm_title(d.get('title', '')) for d in all_downloads}
    
    openreview = OpenReviewScraper(session)
    dl_openreview = openreview.scrape(TARGET_OPENREVIEW, exclude_titles)
    all_downloads.extend([{**d, 'source': 'OpenReview'} for d in dl_openreview])
    
    # 3. Unpaywall
    print(f"\n{'='*60}")
    print("[3/3] Unpaywall")
    print(f"{'='*60}")
    
    # Get additional DOIs from OpenAlex for Unpaywall
    exclude_titles = {norm_title(d.get('title', '')) for d in all_downloads}
    exclude_dois = {(d.get('doi') or '').lower() for d in all_downloads if d.get('doi')}
    
    # Fetch more candidates
    extra_works = []
    for query in QUERIES[5:]:  # Use remaining queries
        works = openalex.fetch_works(query, max_results=200)
        for w in works:
            title = w.get('title', '')
            doi = (w.get('doi') or '').replace('https://doi.org/', '')
            
            if not doi or doi.lower() in exclude_dois:
                continue
            if norm_title(title) in exclude_titles:
                continue
            
            extra_works.append({
                'title': title,
                'abstract': openalex.reconstruct_abstract(w.get('abstract_inverted_index')),
                'date': w.get('publication_date', ''),
                'doi': doi,
            })
    
    unpaywall = UnpaywallScraper(session, UNPAYWALL_EMAIL)
    dl_unpaywall = unpaywall.scrape(extra_works, TARGET_UNPAYWALL)
    all_downloads.extend([{**d, 'source': 'Unpaywall'} for d in dl_unpaywall])
    
    # Save summary
    summary_path = BASE_OUT / "download_summary.csv"
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['title', 'date', 'doi', 'source'])
        writer.writeheader()
        for r in all_downloads:
            writer.writerow({
                'title': r.get('title', '')[:100],
                'date': r.get('date', ''),
                'doi': r.get('doi', ''),
                'source': r.get('source', ''),
            })
    
    elapsed = time.time() - start_time
    
    # Final summary
    print(f"\n{'#'*70}")
    print("# FINAL SUMMARY")
    print(f"{'#'*70}")
    print(f"OpenAlex:    {len(dl_openalex):>4} papers")
    print(f"OpenReview:  {len(dl_openreview):>4} papers")
    print(f"Unpaywall:   {len(dl_unpaywall):>4} papers")
    print(f"{'─'*70}")
    print(f"TOTAL:       {len(all_downloads):>4} papers")
    print(f"Time:        {elapsed/60:.1f} minutes")
    print(f"{'#'*70}")
    print(f"Output: {BASE_OUT}")
    print()

if __name__ == "__main__":
    main()