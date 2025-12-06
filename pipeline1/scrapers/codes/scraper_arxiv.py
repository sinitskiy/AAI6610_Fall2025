#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
arXiv Scraper v2.0 - Optimized for ML Uncertainty Research
- Reads settings from config.yaml
- Better rate limiting & error handling
- Improved search strategy
"""

import os
import sys
import time
import json
import logging
import requests
import feedparser
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set
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
SCRIPT_DIR = Path(__file__).parent.resolve()      # scrapers/codes/
SCRAPERS_DIR = SCRIPT_DIR.parent                   # scrapers/
PIPELINE_DIR = SCRAPERS_DIR.parent                 # pipeline1/
PROJECT_ROOT = PIPELINE_DIR.parent                 # AAI6610_FALL2025/

# Config and output paths
CONFIG_PATH = PIPELINE_DIR / "config.yaml"
ENV_PATH = PIPELINE_DIR / ".env"
OUTPUT_DIR = SCRAPERS_DIR / "outputs" / "arxiv_papers"

# ============================================================================
# Load Environment Variables
# ============================================================================
def load_env():
    """Load .env file"""
    env_paths = [ENV_PATH, PROJECT_ROOT / ".env"]
    
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip())
            logger.info(f"Loaded environment from {env_path.name}")
            return
    
    logger.warning("No .env file found")

load_env()

# ============================================================================
# Load Configuration
# ============================================================================
def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_paths = [CONFIG_PATH, PROJECT_ROOT / "config.yaml"]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded config from {config_path.name}")
                return config
            except Exception as e:
                logger.warning(f"Error loading {config_path}: {e}")
    
    logger.warning("Using default configuration")
    return get_default_config()

def get_default_config() -> dict:
    """Default configuration"""
    return {
        'scrapers': {
            'arxiv': {
                'enabled': True,
                'target_papers': 1000,
                'max_results_per_query': 200,
                'date_range_years': 3,
                'batch_size': 100,
                'rate_limit_delay': 3.0,
                'download_pdfs': True,
            }
        }
    }

# ============================================================================
# Search Configuration
# ============================================================================
# Optimized search terms - grouped by specificity
SEARCH_TERMS_HIGH_PRIORITY = [
    # Title matches (highest relevance)
    'ti:"uncertainty quantification" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'ti:"predictive uncertainty" AND (cat:cs.LG OR cat:cs.AI)',
    'ti:"epistemic uncertainty" AND cat:cs.LG',
    'ti:"aleatoric uncertainty" AND cat:cs.LG',
    'ti:"confidence calibration" AND cat:cs.LG',
    'ti:"bayesian neural network" AND cat:cs.LG',
]

SEARCH_TERMS_MEDIUM_PRIORITY = [
    # Abstract matches (good relevance)
    'abs:"uncertainty estimation" AND cat:cs.LG',
    'abs:"monte carlo dropout" AND cat:cs.LG',
    'abs:"deep ensemble" AND abs:uncertainty AND cat:cs.LG',
    'abs:"conformal prediction" AND cat:cs.LG',
    'abs:"calibration error" AND cat:cs.LG',
    'abs:"out-of-distribution detection" AND cat:cs.LG',
    'abs:"probabilistic prediction" AND cat:cs.LG',
    'abs:"prediction interval" AND cat:cs.LG',
    'abs:"evidential deep learning" AND cat:cs.LG',
    'abs:"temperature scaling" AND cat:cs.LG',
]

SEARCH_TERMS_LOW_PRIORITY = [
    # Broader matches
    'abs:"bayesian deep learning" AND cat:cs.LG',
    'abs:"gaussian process" AND abs:prediction AND cat:cs.LG',
    'abs:"variational inference" AND abs:uncertainty AND cat:cs.LG',
    'abs:"active learning" AND abs:uncertainty AND cat:cs.LG',
    'abs:"selective prediction" AND cat:cs.LG',
    'abs:"distribution shift" AND cat:cs.LG',
]

# Uncertainty keywords for filtering
UNCERTAINTY_KEYWORDS = {
    'high': ['uncertainty quantification', 'uncertainty estimation', 'predictive uncertainty',
             'epistemic uncertainty', 'aleatoric uncertainty', 'confidence calibration',
             'bayesian neural', 'monte carlo dropout', 'deep ensemble'],
    'medium': ['uncertainty', 'confidence', 'calibration', 'bayesian', 'probabilistic',
               'prediction interval', 'credible interval', 'conformal prediction'],
    'low': ['ensemble', 'distribution shift', 'out-of-distribution', 'ood detection']
}

# ============================================================================
# arXiv API Client
# ============================================================================
class ArxivClient:
    """arXiv API client with rate limiting"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, rate_limit_delay: float = 3.0):
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AAI6610-Pipeline/2.0 (ML Uncertainty Research)'
        })
    
    def _wait_for_rate_limit(self):
        """Respect rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def search(self, query: str, start: int = 0, max_results: int = 100) -> List[dict]:
        """Search arXiv API"""
        self._wait_for_rate_limit()
        
        params = {
            'search_query': query,
            'start': start,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending',
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            feed = feedparser.parse(response.text)
            return feed.entries
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
    
    def search_all(self, query: str, max_total: int = 200) -> List[dict]:
        """Search with pagination"""
        all_results = []
        batch_size = 100
        start = 0
        
        while len(all_results) < max_total:
            batch = self.search(query, start, batch_size)
            
            if not batch:
                break
            
            all_results.extend(batch)
            
            if len(batch) < batch_size:
                break
            
            start += batch_size
            logger.debug(f"  Fetched {len(all_results)}/{max_total}")
        
        return all_results[:max_total]
    
    def download_pdf(self, url: str, filepath: Path, timeout: int = 60) -> bool:
        """Download PDF file"""
        try:
            self._wait_for_rate_limit()
            
            response = self.session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except Exception as e:
            logger.debug(f"PDF download failed: {e}")
            return False

# ============================================================================
# Paper Processor
# ============================================================================
class PaperProcessor:
    """Process and filter arXiv papers"""
    
    ML_CATEGORIES = {'cs.LG', 'cs.AI', 'stat.ML', 'cs.CV', 'cs.NE', 'cs.CL'}
    
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
    
    def parse_entry(self, entry: dict) -> Optional[Dict]:
        """Parse arXiv entry into structured format"""
        try:
            arxiv_id = entry.id.split('/')[-1].split('v')[0]  # Remove version
            
            # Parse date
            published_str = entry.get('published', '')
            try:
                published = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ")
            except:
                return None
            
            # Extract categories
            categories = []
            if 'tags' in entry:
                categories = [tag.get('term', '') for tag in entry.tags]
            elif 'arxiv_primary_category' in entry:
                categories = [entry.arxiv_primary_category.get('term', '')]
            
            # Extract PDF URL
            pdf_url = None
            for link in entry.get('links', []):
                if link.get('title') == 'pdf' or link.get('type') == 'application/pdf':
                    pdf_url = link.get('href')
                    break
            
            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            return {
                'arxiv_id': arxiv_id,
                'title': entry.title.replace('\n', ' ').strip(),
                'abstract': entry.summary.replace('\n', ' ').strip(),
                'authors': [a.get('name', '') for a in entry.get('authors', [])],
                'published': published,
                'published_str': published_str,
                'categories': categories,
                'pdf_url': pdf_url,
                'arxiv_url': entry.id,
            }
            
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return None
    
    def is_valid_date(self, paper: dict) -> bool:
        """Check if paper is within date range"""
        pub_date = paper.get('published')
        if not pub_date:
            return False
        return self.start_date <= pub_date <= self.end_date
    
    def is_ml_related(self, paper: dict) -> bool:
        """Check if paper is ML-related"""
        categories = set(paper.get('categories', []))
        return bool(categories & self.ML_CATEGORIES)
    
    def compute_relevance_score(self, paper: dict) -> float:
        """Compute relevance score based on uncertainty keywords"""
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()
        
        score = 0.0
        
        # High priority keywords (weight: 3)
        for kw in UNCERTAINTY_KEYWORDS['high']:
            if kw in text:
                score += 3.0
        
        # Medium priority keywords (weight: 1)
        for kw in UNCERTAINTY_KEYWORDS['medium']:
            if kw in text:
                score += 1.0
        
        # Low priority keywords (weight: 0.5)
        for kw in UNCERTAINTY_KEYWORDS['low']:
            if kw in text:
                score += 0.5
        
        return score
    
    def filter_paper(self, paper: dict, min_score: float = 1.0) -> bool:
        """Filter paper based on all criteria"""
        if not self.is_valid_date(paper):
            return False
        
        if not self.is_ml_related(paper):
            return False
        
        score = self.compute_relevance_score(paper)
        paper['relevance_score'] = score
        
        return score >= min_score

# ============================================================================
# Main Scraper
# ============================================================================
class ArxivScraper:
    """Main arXiv scraper"""
    
    def __init__(self, config: dict):
        scraper_config = config.get('scrapers', {}).get('arxiv', {})
        
        self.target_papers = scraper_config.get('target_papers', 1000)
        self.max_per_query = scraper_config.get('max_results_per_query', 200)
        self.date_range_years = scraper_config.get('date_range_years', 3)
        self.download_pdfs = scraper_config.get('download_pdfs', True)
        rate_limit = scraper_config.get('rate_limit_delay', 3.0)
        
        # Setup dates
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * self.date_range_years)
        
        # Initialize components
        self.client = ArxivClient(rate_limit_delay=rate_limit)
        self.processor = PaperProcessor(self.start_date, self.end_date)
        
        # Tracking
        self.seen_ids: Set[str] = set()
        self.papers: List[dict] = []
        self.stats = {
            'total_fetched': 0,
            'date_filtered': 0,
            'category_filtered': 0,
            'relevance_filtered': 0,
            'duplicates': 0,
            'downloaded': 0,
            'download_failed': 0,
        }
    
    def collect_papers(self) -> List[dict]:
        """Collect papers from all search terms"""
        logger.info(f"Target: {self.target_papers} papers")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        
        # Prioritized search
        all_terms = (
            [(t, 'high') for t in SEARCH_TERMS_HIGH_PRIORITY] +
            [(t, 'medium') for t in SEARCH_TERMS_MEDIUM_PRIORITY] +
            [(t, 'low') for t in SEARCH_TERMS_LOW_PRIORITY]
        )
        
        for idx, (query, priority) in enumerate(all_terms, 1):
            if len(self.papers) >= self.target_papers:
                logger.info(f"Reached target ({self.target_papers} papers)")
                break
            
            logger.info(f"[{idx}/{len(all_terms)}] {query[:60]}...")
            
            entries = self.client.search_all(query, self.max_per_query)
            self.stats['total_fetched'] += len(entries)
            
            added = 0
            for entry in entries:
                paper = self.processor.parse_entry(entry)
                if not paper:
                    continue
                
                # Check duplicate
                if paper['arxiv_id'] in self.seen_ids:
                    self.stats['duplicates'] += 1
                    continue
                
                # Filter
                if not self.processor.is_valid_date(paper):
                    self.stats['date_filtered'] += 1
                    continue
                
                if not self.processor.is_ml_related(paper):
                    self.stats['category_filtered'] += 1
                    continue
                
                score = self.processor.compute_relevance_score(paper)
                min_score = 3.0 if priority == 'high' else 1.5 if priority == 'medium' else 1.0
                
                if score < min_score:
                    self.stats['relevance_filtered'] += 1
                    continue
                
                paper['relevance_score'] = score
                self.seen_ids.add(paper['arxiv_id'])
                self.papers.append(paper)
                added += 1
            
            logger.info(f"  Added: {added} | Total: {len(self.papers)}")
        
        # Sort by relevance
        self.papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return self.papers[:self.target_papers]
    
    def save_papers(self, output_dir: Path) -> int:
        """Save papers to output directory"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved = 0
        
        for paper in self.papers:
            arxiv_id = paper['arxiv_id']
            
            # Save metadata as TXT
            txt_path = output_dir / f"{arxiv_id}.txt"
            
            metadata = [
                f"Title: {paper['title']}",
                f"Authors: {', '.join(paper['authors'][:5])}",
                f"Published: {paper['published_str']}",
                f"arXiv ID: {arxiv_id}",
                f"Categories: {', '.join(paper['categories'])}",
                f"Relevance Score: {paper.get('relevance_score', 0):.1f}",
                f"URL: {paper['arxiv_url']}",
                "",
                "Abstract:",
                paper['abstract'],
            ]
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(metadata))
            
            # Download PDF (optional)
            if self.download_pdfs:
                pdf_path = output_dir / f"{arxiv_id}.pdf"
                
                if not pdf_path.exists():
                    if self.client.download_pdf(paper['pdf_url'], pdf_path):
                        self.stats['downloaded'] += 1
                    else:
                        self.stats['download_failed'] += 1
                else:
                    self.stats['downloaded'] += 1
            
            saved += 1
            
            if saved % 50 == 0:
                logger.info(f"  Saved: {saved}/{len(self.papers)}")
        
        return saved
    
    def save_metadata(self, output_dir: Path):
        """Save full metadata as JSON"""
        metadata = []
        
        for paper in self.papers:
            metadata.append({
                'arxiv_id': paper['arxiv_id'],
                'title': paper['title'],
                'abstract': paper['abstract'],
                'authors': paper['authors'],
                'published': paper['published_str'],
                'categories': paper['categories'],
                'relevance_score': paper.get('relevance_score', 0),
            })
        
        json_path = output_dir / 'papers_metadata.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved metadata: {json_path.name}")
    
    def print_stats(self):
        """Print collection statistics"""
        print(f"\n{'='*60}")
        print("COLLECTION STATISTICS")
        print(f"{'='*60}")
        print(f"Total fetched:      {self.stats['total_fetched']:>6}")
        print(f"Duplicates:         {self.stats['duplicates']:>6}")
        print(f"Date filtered:      {self.stats['date_filtered']:>6}")
        print(f"Category filtered:  {self.stats['category_filtered']:>6}")
        print(f"Relevance filtered: {self.stats['relevance_filtered']:>6}")
        print(f"{'─'*60}")
        print(f"Final papers:       {len(self.papers):>6}")
        
        if self.download_pdfs:
            print(f"PDFs downloaded:    {self.stats['downloaded']:>6}")
            print(f"Download failed:    {self.stats['download_failed']:>6}")
        
        print(f"{'='*60}\n")

# ============================================================================
# Main
# ============================================================================
def main():
    print(f"\n{'#'*70}")
    print(f"# arXiv Scraper v2.0")
    print(f"# ML Uncertainty Quantification Research")
    print(f"{'#'*70}\n")
    
    # Load configuration
    config = load_config()
    
    # Check if enabled
    if not config.get('scrapers', {}).get('arxiv', {}).get('enabled', True):
        logger.warning("arXiv scraper is disabled in config.yaml")
        logger.info("Set scrapers.arxiv.enabled: true to enable")
        return
    
    # Initialize scraper
    scraper = ArxivScraper(config)
    
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Collect papers
    papers = scraper.collect_papers()
    
    if not papers:
        logger.error("No papers collected!")
        return
    
    # Save papers
    logger.info(f"\nSaving {len(papers)} papers...")
    saved = scraper.save_papers(OUTPUT_DIR)
    scraper.save_metadata(OUTPUT_DIR)
    
    # Print statistics
    scraper.print_stats()
    
    print(f"Output: {OUTPUT_DIR}")
    print(f"TXT files: {len(list(OUTPUT_DIR.glob('*.txt')))}")
    print(f"PDF files: {len(list(OUTPUT_DIR.glob('*.pdf')))}")
    print()

if __name__ == "__main__":
    main()