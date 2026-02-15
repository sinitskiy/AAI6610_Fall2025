#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bioRxiv Crawler - Complete Balanced Edition
Monthly search + synonym matching + target 3000+ papers
"""

import os
import re
import time
import random
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

# PDF parsing
try:
    from pypdf import PdfReader
    PYPDF_OK = True
except:
    PYPDF_OK = False
    print("pypdf not available")

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    PDFMINER_OK = True
except:
    PDFMINER_OK = False
    print("pdfminer not available")

# ========== Path Configuration ==========
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRAPER_OUTPUTS = PROJECT_ROOT / "scrapers" / "outputs"
OUTPUT_DIR = SCRAPER_OUTPUTS / "biorxiv_papers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ==============================

# ============ Configuration (Balanced Version) ============
QUERY = "uncertainty prediction machine learning"
START_YEAR = 2020  # 10 years of data (2015-2025)
MAX_PAPERS = 3000  # Target 3000 papers
MIN_CATEGORIES = 2 # At least match 1 category
MIN_SCORE = 5      # Minimum score

class BioRxivDownloader:
    """bioRxiv downloader - Complete balanced version"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_base = "https://api.biorxiv.org"
        self.web_base = "https://www.biorxiv.org"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Synonym dictionary (100+ synonyms)
        self.synonyms = {
            'uncertainty': [
                'uncertainty', 'uncertainties', 'uncertain',
                'ambiguity', 'ambiguous',
                'variability', 'variance', 'variation',
                'noise', 'error', 'errors',
                'stochastic', 'random', 'randomness'
            ],
            'probabilistic': [
                'probabilistic', 'probability', 'probable',
                'likelihood', 'maximum likelihood',
                'distribution', 'distributions',
                'gaussian', 'normal distribution',
                'beta distribution', 'dirichlet'
            ],
            'bayesian': [
                'bayesian', 'bayes', 'bayesian inference',
                'posterior', 'prior', 'posterior distribution',
                'prior distribution', 'bayesian network',
                'mcmc', 'monte carlo', 'markov chain',
                'gibbs sampling', 'metropolis'
            ],
            'confidence': [
                'confidence', 'confidence interval', 'confident',
                'credible interval', 'prediction interval',
                'tolerance interval', 'uncertainty bounds',
                'error bars', 'confidence level'
            ],
            'calibration': [
                'calibration', 'calibrated', 'calibrate',
                'reliability', 'reliable', 'well-calibrated',
                'temperature scaling', 'platt scaling',
                'calibration error', 'ece', 'expected calibration',
                'isotonic regression'
            ],
            'prediction': [
                'prediction', 'predictions', 'predictive', 'predict',
                'forecasting', 'forecast', 'forecasts',
                'prognosis', 'prognostic',
                'estimation', 'estimate', 'estimating', 'estimator',
                'inference', 'infer', 'inferring'
            ],
            'machine_learning': [
                'machine learning', 'deep learning',
                'neural network', 'neural net', 'nn', 'dnn',
                'cnn', 'rnn', 'lstm', 'gru', 'transformer',
                'artificial intelligence', 'ai',
                'supervised learning', 'unsupervised learning',
                'reinforcement learning', 'transfer learning'
            ],
            'model': [
                'model', 'models', 'modeling', 'modelling',
                'algorithm', 'algorithms', 'algorithmic',
                'classifier', 'classification',
                'regression', 'regressor',
                'predictor', 'predictive model'
            ],
            'ensemble': [
                'ensemble', 'ensembles', 'ensemble method',
                'bootstrap', 'bootstrapping',
                'bagging', 'boosting',
                'random forest', 'gradient boosting',
                'dropout', 'monte carlo dropout',
                'deep ensemble'
            ],
            'uncertainty_types': [
                'epistemic', 'aleatoric',
                'model uncertainty', 'data uncertainty',
                'parameter uncertainty', 'structural uncertainty',
                'epistemic uncertainty', 'aleatoric uncertainty',
                'heteroscedastic', 'homoscedastic'
            ]
        }
    
    def search_by_date_range(self, start_date: datetime, end_date: datetime) -> List[dict]:
        """Search by date range"""
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        url = f"{self.api_base}/details/biorxiv/{start_str}/{end_str}"
        
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if 'collection' in data:
                return data['collection']
            return []
        except Exception:
            return []
    
    def search_all_years(self, start_year: int) -> List[dict]:
        """Search all years by month - bypass 100 paper limit"""
        print(f"Searching bioRxiv from {start_year} to {datetime.now().year}")
        print(f"Strategy: Monthly queries to bypass API limit\n")
        
        all_papers = []
        total_requests = 0
        
        for year in range(start_year, datetime.now().year + 1):
            print(f"  Year {year}:")
            year_papers = 0
            
            for month in range(1, 13):
                month_start = datetime(year, month, 1)
                
                # First day of next month
                if month == 12:
                    month_end = datetime(year + 1, 1, 1)
                else:
                    month_end = datetime(year, month + 1, 1)
                
                # Don't exceed today
                if month_start > datetime.now():
                    break
                
                # Search this month
                papers = self.search_by_date_range(month_start, month_end)
                
                if papers:
                    all_papers.extend(papers)
                    year_papers += len(papers)
                    print(f"    Month {month:>2}: {len(papers):>3} papers")
                
                total_requests += 1
                time.sleep(0.5)  # API-friendly delay
            
            print(f"    -> Year {year} total: {year_papers:>4} papers\n")
        
        print(f"{'='*70}")
        print(f"Total collected: {len(all_papers)} papers")
        print(f"Total API calls: {total_requests}")
        print(f"Average per call: {len(all_papers)/total_requests:.1f} papers")
        print(f"{'='*70}\n")
        
        return all_papers
    
    def match_synonyms(self, text: str, category: str) -> int:
        """Match synonym count"""
        if category not in self.synonyms:
            return 0
        
        count = 0
        for synonym in self.synonyms[category]:
            if synonym in text:
                count += 1
        
        return count
    
    def filter_papers(self, papers: List[dict], query: str) -> List[dict]:
        """Synonym-based intelligent filtering - balanced version"""
        print(f"Applying synonym-based filtering...")
        print(f"Min categories: {MIN_CATEGORIES}, Min score: {MIN_SCORE}\n")
        
        # Light exclusion (only exclude most obviously irrelevant)
        exclude_kw = [
            'phase iii clinical trial',
            'randomized double-blind placebo-controlled',
            'patient survival outcome analysis',
            'x-ray crystal structure refinement',
            'cryo-electron microscopy structure'
        ]
        
        filtered = []
        
        for paper in papers:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            combined = f"{title} {abstract}"
            
            # Exclude
            if any(ex in combined for ex in exclude_kw):
                continue
            
            # Synonym matching
            scores = {
                'uncertainty': self.match_synonyms(combined, 'uncertainty'),
                'probabilistic': self.match_synonyms(combined, 'probabilistic'),
                'bayesian': self.match_synonyms(combined, 'bayesian'),
                'confidence': self.match_synonyms(combined, 'confidence'),
                'calibration': self.match_synonyms(combined, 'calibration'),
                'prediction': self.match_synonyms(combined, 'prediction'),
                'ml': self.match_synonyms(combined, 'machine_learning'),
                'model': self.match_synonyms(combined, 'model'),
                'ensemble': self.match_synonyms(combined, 'ensemble'),
                'unc_types': self.match_synonyms(combined, 'uncertainty_types')
            }
            
            # At least N categories matched
            categories_matched = sum(1 for v in scores.values() if v > 0)
            
            if categories_matched < MIN_CATEGORIES:
                continue
            
            # Weighted total score
            total_score = (
                scores['uncertainty'] * 3 +
                scores['probabilistic'] * 2 +
                scores['bayesian'] * 2 +
                scores['confidence'] * 2 +
                scores['calibration'] * 2 +
                scores['prediction'] * 2 +
                scores['ml'] * 2 +
                scores['model'] * 1 +
                scores['ensemble'] * 2 +
                scores['unc_types'] * 3
            )
            
            # Title bonus
            title_kw = ['uncertainty', 'prediction', 'bayesian', 'probabilistic', 'confidence', 'model', 'learning']
            if any(kw in title for kw in title_kw):
                total_score += 5
            
            # Score threshold
            if total_score >= MIN_SCORE:
                paper['_score'] = total_score
                paper['_categories'] = categories_matched
                paper['_details'] = scores
                filtered.append(paper)
        
        # Sort
        filtered.sort(key=lambda p: p.get('_score', 0), reverse=True)
        
        print(f"Matched: {len(filtered)} papers ({len(filtered)/len(papers)*100:.1f}%)")
        
        if filtered:
            high = sum(1 for p in filtered if p['_score'] > 30)
            medium = sum(1 for p in filtered if 15 <= p['_score'] <= 30)
            low = sum(1 for p in filtered if 5 <= p['_score'] < 15)
            vlow = sum(1 for p in filtered if p['_score'] < 5)
            
            print(f"\nScore distribution:")
            print(f"  High (>30):     {high:>4} papers")
            print(f"  Medium (15-30): {medium:>4} papers")
            print(f"  Low (5-15):     {low:>4} papers")
            print(f"  Very low (<5):  {vlow:>4} papers")
            
            print(f"\nTop 15 highest scoring:")
            for i, p in enumerate(filtered[:15], 1):
                print(f"  {i:>2}. Score={p['_score']:>3} | {p.get('title', '')[:60]}...")
        
        print()
        return filtered
    
    def download_pdf(self, doi: str, version: str = "1") -> Optional[Path]:
        """Download PDF"""
        pdf_urls = [
            f"{self.web_base}/content/{doi}v{version}.full.pdf",
            f"{self.web_base}/content/{doi}.full.pdf",
        ]
        
        for pdf_url in pdf_urls:
            try:
                resp = self.session.get(pdf_url, timeout=60)
                
                if resp.status_code == 200 and 'pdf' in resp.headers.get('content-type', '').lower():
                    doi_clean = doi.replace('/', '_').replace('.', '_')
                    pdf_path = self.output_dir / f"{doi_clean}.pdf"
                    
                    with open(pdf_path, 'wb') as f:
                        f.write(resp.content)
                    
                    return pdf_path
            except:
                continue
        
        return None
    
    def extract_pdf_text(self, pdf_path: Path, max_pages: int = 20) -> str:
        """Extract text from PDF"""
        text = ""
        
        # Method 1: pypdf
        if PYPDF_OK:
            try:
                reader = PdfReader(str(pdf_path), strict=False)
                n_pages = min(len(reader.pages), max_pages) if max_pages else len(reader.pages)
                
                for i in range(n_pages):
                    page_text = reader.pages[i].extract_text() or ""
                    text += page_text + "\n"
                
                if len(text.strip()) > 100:
                    return text.strip()
            except:
                pass
        
        # Method 2: pdfminer (fallback)
        if PDFMINER_OK and not text:
            try:
                kwargs = {'maxpages': max_pages} if max_pages else {}
                text = pdfminer_extract(str(pdf_path), **kwargs)
                if text:
                    return text.strip()
            except:
                pass
        
        return text
    
    def save_full_txt(self, paper: dict, pdf_path: Optional[Path]) -> bool:
        """Save complete TXT (metadata + abstract + PDF full text)"""
        doi = paper.get('doi', 'unknown')
        doi_clean = doi.replace('/', '_').replace('.', '_')
        txt_path = self.output_dir / f"{doi_clean}.txt"
        
        lines = [
            f"Title: {paper.get('title', 'N/A')}",
            f"Date: {paper.get('date', 'N/A')}",
            f"DOI: {doi}",
            f"Source: bioRxiv",
            f"Match Score: {paper.get('_score', 0)}",
            f"Categories Matched: {paper.get('_categories', 0)}",
            "",
            "Abstract:",
            paper.get('abstract', 'N/A'),
            "",
            "="*80,
            ""
        ]
        
        # Extract PDF full text
        if pdf_path and pdf_path.exists():
            pdf_text = self.extract_pdf_text(pdf_path, max_pages=20)
            
            if pdf_text and len(pdf_text) > 100:
                lines.append("Full Text:")
                lines.append(pdf_text)
            else:
                lines.append("Full Text: (PDF extraction failed)")
        else:
            lines.append("Full Text: (PDF not available)")
        
        # Save
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return True
        except Exception:
            return False
    
    def run(self, query: str, start_year: int, max_papers: int):
        """Run complete pipeline"""
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print("bioRxiv Crawler - Complete Balanced Edition")
        print(f"{'='*70}")
        print(f"Query        : {query}")
        print(f"Years        : {start_year}-{datetime.now().year} ({datetime.now().year - start_year + 1} years)")
        print(f"Target       : up to {max_papers} papers")
        print(f"Strategy     : Monthly queries + Synonym matching")
        print(f"Min score    : {MIN_SCORE}")
        print(f"Min categories: {MIN_CATEGORIES}")
        print(f"Output       : {self.output_dir}")
        print(f"{'='*70}\n")
        
        # Step 1: Monthly search
        print(f"STEP 1: SEARCHING")
        print(f"{'─'*70}")
        all_papers = self.search_all_years(start_year)
        
        if not all_papers:
            print("No papers found")
            return
        
        # Step 2: Synonym filtering
        print(f"STEP 2: FILTERING")
        print(f"{'─'*70}")
        filtered = self.filter_papers(all_papers, query)
        
        if not filtered:
            print("No papers matched filters")
            return
        
        # Limit quantity
        if len(filtered) > max_papers:
            print(f"Limiting to top {max_papers} papers by score\n")
            filtered = filtered[:max_papers]
        
        # Step 3: Download
        print(f"STEP 3: DOWNLOADING")
        print(f"{'─'*70}")
        print(f"Processing {len(filtered)} papers...\n")
        
        success_pdf = 0
        success_txt = 0
        failed_pdf = 0
        
        for i, paper in enumerate(filtered, 1):
            title = paper.get('title', 'Unknown')[:50]
            score = paper.get('_score', 0)
            
            print(f"[{i:>4}/{len(filtered)}] Score={score:>3}: {title}...")
            
            # Download PDF
            pdf_path = self.download_pdf(paper.get('doi', ''), paper.get('version', '1'))
            
            if pdf_path:
                print(f"    PDF: OK", end="")
                success_pdf += 1
            else:
                print(f"    PDF: FAILED", end="")
                failed_pdf += 1
            
            # Save TXT
            if self.save_full_txt(paper, pdf_path):
                print(f" | TXT: OK")
                success_txt += 1
            else:
                print(f" | TXT: FAILED")
            
            # Progress report
            if i % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(filtered) - i) * avg_time
                
                print(f"\n  {'─'*60}")
                print(f"  Progress: {i}/{len(filtered)} ({i/len(filtered)*100:.1f}%)")
                print(f"  PDFs: {success_pdf} | Failed: {failed_pdf} | TXTs: {success_txt}")
                print(f"  Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min")
                print(f"  {'─'*60}\n")
            
            time.sleep(random.uniform(2, 4))
        
        # Final statistics
        elapsed_total = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"Time range       : {start_year}-{datetime.now().year}")
        print(f"Papers searched  : {len(all_papers)}")
        print(f"Papers matched   : {len(filtered)} ({len(filtered)/len(all_papers)*100:.1f}%)")
        print(f"Papers processed : {len(filtered)}")
        print(f"{'─'*70}")
        print(f"PDFs downloaded  : {success_pdf} ({success_pdf/len(filtered)*100:.1f}%)")
        print(f"PDFs failed      : {failed_pdf} ({failed_pdf/len(filtered)*100:.1f}%)")
        print(f"TXTs created     : {success_txt} ({success_txt/len(filtered)*100:.1f}%)")
        print(f"{'─'*70}")
        print(f"Time elapsed     : {elapsed_total/60:.1f} minutes ({elapsed_total/3600:.2f} hours)")
        print(f"Avg per paper    : {elapsed_total/len(filtered):.1f} seconds")
        print(f"{'─'*70}")
        print(f"Output directory : {self.output_dir}")
        print(f"PDF files        : {len(list(self.output_dir.glob('*.pdf')))}")
        print(f"TXT files        : {len(list(self.output_dir.glob('*.txt')))}")
        print(f"{'='*70}\n")
        
        # Save statistics
        stats = {
            "query": query,
            "years": f"{start_year}-{datetime.now().year}",
            "total_searched": len(all_papers),
            "total_matched": len(filtered),
            "match_rate": f"{len(filtered)/len(all_papers)*100:.1f}%",
            "pdfs_downloaded": success_pdf,
            "pdfs_failed": failed_pdf,
            "txts_created": success_txt,
            "time_minutes": round(elapsed_total/60, 1)
        }
        
        import json
        with open(self.output_dir / "scraping_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Stats saved to: {self.output_dir / 'scraping_stats.json'}\n")


def main():
    """Main program"""
    print(f"\n{'='*70}")
    print("PDF Parser Status:")
    print(f"{'='*70}")
    print(f"pypdf    : {'Available' if PYPDF_OK else 'Not available'}")
    print(f"pdfminer : {'Available' if PDFMINER_OK else 'Not available'}")
    
    if not PYPDF_OK and not PDFMINER_OK:
        print(f"\nWARNING: No PDF parser available!")
        print(f"   Install: pip install pypdf pdfminer.six")
        print(f"   Will only save metadata + abstract\n")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    print(f"{'='*70}\n")
    
    downloader = BioRxivDownloader(OUTPUT_DIR)
    downloader.run(QUERY, START_YEAR, MAX_PAPERS)


if __name__ == "__main__":
    main()
