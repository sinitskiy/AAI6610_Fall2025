#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
arXiv Crawler - Only responsible for downloading papers
Adapted to whole_pipeline project
"""

import os
import requests
import feedparser
from datetime import datetime
import json
import time
from pathlib import Path

# ========== Path Configuration ==========
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRAPER_OUTPUTS = PROJECT_ROOT / "scrapers" / "outputs"
OUTPUT_DIR = SCRAPER_OUTPUTS / "arxiv_papers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ==============================

# -----------------------------
# Configuration
# -----------------------------
# arXiv search terms
search_terms = [
    'ti:"uncertainty quantification" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'ti:"predictive uncertainty" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'ti:"epistemic uncertainty" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'ti:"aleatoric uncertainty" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"uncertainty estimation" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"confidence calibration" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"bayesian deep learning" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"probabilistic prediction" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"uncertainty aware" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"monte carlo dropout" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"ensemble uncertainty" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"prediction intervals" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"conformal prediction" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'ti:"uncertainty propagation" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'ti:"prediction uncertainty" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'ti:"model uncertainty" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"deep ensembles" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"evidential deep learning" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"prior networks" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"posterior networks" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"normalizing flows" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"temperature scaling" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"expected calibration error" AND cat:cs.LG',
    'abs:"maximum mean calibration error" AND cat:cs.LG',
    'abs:"reliable confidence" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"selective prediction" AND cat:cs.LG',
    'abs:"predictive entropy" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"mutual information" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"BALD" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"active learning" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"OOD detection" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"out-of-distribution" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"distributional shift" AND (cat:cs.LG OR cat:stat.ML)',
    'abs:"covariate shift" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"label shift" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"domain shift" AND abs:"prediction" AND cat:cs.LG',
    'abs:"test-time adaptation" AND cat:cs.LG',
    'abs:"prediction sets" AND (cat:cs.LG OR cat:stat.ML)',
    'abs:"coverage guarantee" AND (cat:cs.LG OR cat:stat.ML)',
    'abs:"conditional coverage" AND cat:stat.ML',
    'abs:"distribution-free uncertainty" AND cat:cs.LG',
    'abs:"probabilistic forecasting" AND cat:cs.LG',
    'abs:"interval prediction" AND cat:cs.LG',
    'abs:"quantile forecasting" AND (cat:cs.LG OR cat:stat.ML)',
    'abs:"variational inference" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"gaussian processes" AND abs:"prediction" AND cat:cs.LG',
    'abs:"posterior sampling" AND (cat:cs.LG OR cat:stat.ML)',
    'abs:"credible intervals" AND (cat:cs.LG OR cat:stat.ML)',
    'abs:"prediction consistency" AND cat:cs.LG',
    'abs:"model confidence" AND cat:cs.LG',
    'abs:"uncertainty decomposition" AND cat:cs.LG',
    'abs:"heteroscedastic uncertainty" AND cat:cs.LG',
    'abs:"homoscedastic uncertainty" AND cat:cs.LG',
    'abs:"uncertainty budget" AND cat:cs.LG',
    'abs:"reliability diagram" AND cat:cs.LG',
    'abs:"calibration error" AND cat:cs.LG',
    'abs:"Platt scaling" AND cat:cs.LG',
    'abs:"isotonic regression" AND abs:"calibration" AND cat:cs.LG',
    'abs:"histogram binning" AND abs:"calibration" AND cat:cs.LG',
]

# Date range
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 10, 15)

# arXiv API
base_url = "http://export.arxiv.org/api/query"
TARGET_PAPERS = 1500

# -----------------------------
# Functions
# -----------------------------
def fetch_papers_batch(query, start=0, max_results=100):
    """Fetch a batch of papers from arXiv API"""
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        feed = feedparser.parse(response.text)
        return feed.entries
    except Exception as e:
        print(f"  Error: {e}")
        return []

def fetch_all_papers_for_query(query, max_total=300):
    """Fetch papers with pagination"""
    all_papers = []
    batch_size = 100
    start = 0
    
    while len(all_papers) < max_total:
        print(f"    Batch {start}...")
        batch = fetch_papers_batch(query, start, batch_size)
        
        if not batch:
            break
        
        all_papers.extend(batch)
        
        if len(batch) < batch_size:
            break
        
        start += batch_size
        time.sleep(3)
    
    return all_papers[:max_total]

def is_in_date_range(date_str):
    """Check if date is within range"""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        return START_DATE <= date <= END_DATE
    except:
        return False

def is_ml_related(entry):
    """Check if ML related"""
    categories = entry.get('tags', [])
    if not categories:
        categories = entry.get('arxiv_primary_category', {}).get('term', '')
        if categories:
            categories = [{'term': categories}]
    
    ml_categories = ['cs.LG', 'cs.AI', 'stat.ML', 'cs.CV', 'cs.NE', 'cs.CL']
    
    for cat in categories:
        if any(ml_cat in cat.get('term', '') for ml_cat in ml_categories):
            return True
    return False

def contains_uncertainty_keywords(title, abstract):
    """Check if contains uncertainty keywords"""
    text = (title + " " + abstract).lower()
    keywords = [
        'uncertainty', 'confidence', 'calibration', 'bayesian',
        'probabilistic', 'ensemble', 'prediction interval',
        'epistemic', 'aleatoric', 'monte carlo dropout',
        'variational inference', 'gaussian process', 'conformal',
        'credible interval', 'quantile regression', 'risk assessment',
        'deep ensemble', 'evidential', 'temperature scaling',
        'expected calibration error', 'reliability', 'coverage',
        'out-of-distribution', 'ood', 'distribution shift',
        'selective prediction', 'predictive entropy', 'bald',
        'active learning', 'test-time', 'prediction set',
        'platt scaling', 'isotonic regression', 'histogram binning'
    ]
    return any(keyword in text for keyword in keywords)

def download_file(url, filepath):
    """Download file"""
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    Download error: {e}")
        return False

# -----------------------------
# Main Program
# -----------------------------
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"arXiv Crawler")
    print(f"{'='*70}")
    print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Target: {TARGET_PAPERS} papers")
    print(f"Search terms: {len(search_terms)}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    # Collect all papers
    all_papers_dict = {}
    
    for i, query in enumerate(search_terms, 1):
        print(f"\n[{i}/{len(search_terms)}] {query[:70]}...")
        papers = fetch_all_papers_for_query(query, max_total=300)
        
        for entry in papers:
            arxiv_id = entry.id.split("/")[-1]
            if arxiv_id not in all_papers_dict:
                all_papers_dict[arxiv_id] = entry
        
        print(f"  Found {len(papers)} | Total unique: {len(all_papers_dict)}")

    print(f"\n{'='*70}")
    print(f"Total unique papers: {len(all_papers_dict)}")
    print(f"{'='*70}\n")

    # Filter and download
    downloaded_count = 0
    papers_metadata = []
    date_filtered = 0
    category_filtered = 0
    keyword_filtered = 0

    for arxiv_id, entry in all_papers_dict.items():
        if downloaded_count >= TARGET_PAPERS:
            print(f"\nReached target: {TARGET_PAPERS}")
            break
        
        # Filter
        if not is_in_date_range(entry.published):
            date_filtered += 1
            continue
        
        if not is_ml_related(entry):
            category_filtered += 1
            continue
        
        title = entry.title.replace("\n", " ").strip()
        abstract = entry.summary.replace("\n", " ").strip()
        
        if not contains_uncertainty_keywords(title, abstract):
            keyword_filtered += 1
            continue
        
        # Download
        published = entry.published
        pdf_url = None
        doi = entry.get("arxiv_doi", entry.get("id"))
        
        categories = []
        if 'tags' in entry:
            categories = [tag['term'] for tag in entry.tags]
        
        for link in entry.links:
            if link.rel == "related" and "doi.org" in link.href:
                doi = link.href
            if link.get("title") == "pdf":
                pdf_url = link.href
        
        if not pdf_url:
            continue
        
        pdf_filename = OUTPUT_DIR / f"{arxiv_id}.pdf"
        txt_filename = OUTPUT_DIR / f"{arxiv_id}.txt"
        
        # Download PDF
        if not pdf_filename.exists():
            print(f"[{downloaded_count + 1}/{TARGET_PAPERS}] {title[:60]}...")
            if not download_file(pdf_url, pdf_filename):
                continue
            time.sleep(2)
        
        # Save metadata
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\n")
            f.write(f"Published: {published}\n")
            f.write(f"DOI/ID: {doi}\n")
            f.write(f"Categories: {', '.join(categories)}\n")
            f.write(f"Abstract:\n{abstract}\n")
        
        papers_metadata.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "published": published,
            "categories": categories
        })
        
        downloaded_count += 1
        
        if downloaded_count % 50 == 0:
            print(f"  Progress: {downloaded_count}/{TARGET_PAPERS}")
        
        if downloaded_count % 100 == 0:
            # Save intermediate results
            with open(OUTPUT_DIR / "papers_metadata_temp.json", "w", encoding="utf-8") as f:
                json.dump(papers_metadata, f, ensure_ascii=False, indent=2)

    # Statistics
    print(f"\n{'='*70}")
    print("FILTERING SUMMARY")
    print(f"{'='*70}")
    print(f"Total found: {len(all_papers_dict)}")
    print(f"Filtered by date: {date_filtered}")
    print(f"Filtered by category: {category_filtered}")
    print(f"Filtered by keywords: {keyword_filtered}")
    print(f"Successfully downloaded: {downloaded_count}")
    print(f"{'='*70}\n")

    # Save final metadata
    with open(OUTPUT_DIR / "papers_metadata.json", "w", encoding="utf-8") as f:
        json.dump(papers_metadata, f, ensure_ascii=False, indent=2)

    # Delete temporary file
    temp_file = OUTPUT_DIR / "papers_metadata_temp.json"
    if temp_file.exists():
        temp_file.unlink()

    print(f"arXiv scraping completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"PDF files: {len(list(OUTPUT_DIR.glob('*.pdf')))}")
    print(f"TXT files: {len(list(OUTPUT_DIR.glob('*.txt')))}\n")
