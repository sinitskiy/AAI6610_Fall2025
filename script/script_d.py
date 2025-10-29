#!/usr/bin/env python3

import requests
import json
import os
import time
import random
from datetime import datetime, timedelta
from urllib.parse import urljoin, quote
import re
from pathlib import Path
from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not available.")

class EnhancedBioRxivDownloader:
    def __init__(self, output_dir="biorxiv_papers"):
        self.base_url = "https://api.biorxiv.org"
        self.web_base_url = "https://www.biorxiv.org"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.driver = None
    
    def _random_sleep(self, min_seconds=3, max_seconds=15, reason="general"):
        """Random sleep to mimic human behavior"""
        sleep_ranges = {
            "search": (3, 8),
            "download": (5, 12),
            "manual": (20, 30),
            "rate_limit": (3, 15),
            "general": (min_seconds, max_seconds)
        }
        
        min_s, max_s = sleep_ranges.get(reason, (min_seconds, max_seconds))
        sleep_time = random.uniform(min_s, max_s)
        print(f"Waiting {sleep_time:.1f} seconds...")
        time.sleep(sleep_time)
    
    def _create_safe_filename(self, title):
        """Create safe filename from title"""
        safe_title = re.sub(r'[^\w\s-]', '', title)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)[:100]
        return safe_title.strip('_')
    
    def _get_base_filename(self, paper):
        """Generate consistent base filename for both PDF and metadata"""
        title = paper.get('title', 'unknown')
        safe_title = self._create_safe_filename(title)
        
        doi = paper.get('doi', 'unknown')
        doi_parts = doi.split('/')[-1].split('.')
        doi_number = doi_parts[-1] if doi_parts else 'unknown'
        
        return f"{safe_title}_{doi_number}"
    
    def search_papers_api(self, query="uncertainty prediction ML", days_back=200, max_results=50, start_year=None):
        """Search for papers using bioRxiv API"""
        print(f"API Search: '{query}'")
        
        end_date = datetime.now()
        
        if start_year:
            start_date = datetime(start_year, 1, 1)
            print(f"Date range: {start_year}-01-01 to {end_date.strftime('%Y-%m-%d')}")
            
            all_papers = []
            current_year = start_year
            current_end = datetime.now().year
            
            while current_year <= current_end:
                year_start = datetime(current_year, 1, 1)
                year_end = datetime(current_year, 12, 31) if current_year < current_end else end_date
                
                print(f"Searching year {current_year}...")
                papers = self._search_api_range(year_start, year_end)
                
                if papers:
                    all_papers.extend(papers)
                    print(f"Found {len(papers)} papers in {current_year}")
                
                current_year += 1
            
            papers = all_papers
        else:
            start_date = end_date - timedelta(days=days_back)
            papers = self._search_api_range(start_date, end_date)
        
        if not papers:
            print("No papers found")
            return []
        
        print(f"Total papers retrieved: {len(papers)}")
        
        # Filter papers
        filtered_papers = []
        query_terms = [term.lower() for term in query.split()]
        
        synonyms = {
            'uncertainty': ['uncertainty', 'uncertainties', 'probabilistic', 'confidence', 'bayesian'],
            'prediction': ['prediction', 'predictions', 'predicting', 'predict', 'predictive', 'forecast'],
            'ml': ['machine learning', 'deep learning', 'neural network', 'ml', 'ai'],
            'machine': ['machine learning', 'deep learning'],
            'learning': ['machine learning', 'deep learning', 'learning']
        }
        
        for paper in papers:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            combined_text = f"{title} {abstract}"
            
            direct_matches = sum(1 for term in query_terms if term in combined_text)
            
            synonym_matches = 0
            for term in query_terms:
                if term in synonyms:
                    if any(syn in combined_text for syn in synonyms[term]):
                        synonym_matches += 1
                elif term in combined_text:
                    synonym_matches += 1
            
            final_score = max(direct_matches, synonym_matches) / len(query_terms) if query_terms else 0
            
            if final_score >= 0.50:
                paper['_match_score'] = final_score
                filtered_papers.append(paper)
        
        filtered_papers.sort(key=lambda p: p.get('_match_score', 0), reverse=True)
        filtered_papers = filtered_papers[:max_results]
        
        print(f"Found {len(filtered_papers)} matching papers")
        return filtered_papers
    
    def _search_api_range(self, start_date, end_date):
        """Helper function to search a specific date range"""
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        search_url = f"{self.base_url}/details/biorxiv/{start_date_str}/{end_date_str}"
        
        try:
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'collection' in data:
                return data['collection']
            return []
        except:
            return []
    
    def download_pdf_direct(self, paper):
        """Direct PDF download"""
        doi = paper.get('doi')
        if not doi:
            return None
        
        pdf_urls = [
            f"{self.web_base_url}/content/{doi}v{paper.get('version', '1')}.full.pdf",
            f"{self.web_base_url}/content/{doi}.full.pdf",
        ]
        
        for pdf_url in pdf_urls:
            try:
                response = self.session.get(pdf_url, timeout=60)
                
                if response.status_code == 200 and 'pdf' in response.headers.get('content-type', '').lower():
                    # Save with original filename from URL
                    filename = pdf_url.split('/')[-1]
                    filepath = self.output_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {filename}")
                    return filepath
            except:
                continue
        
        print(f"Download failed")
        return None
    
    def setup_selenium_driver(self):
        """Setup Selenium driver"""
        if not SELENIUM_AVAILABLE:
            return False
        
        try:
            options = webdriver.ChromeOptions()
            prefs = {
                "download.default_directory": str(self.output_dir.absolute()),
                "download.prompt_for_download": False,
                "plugins.always_open_pdf_externally": True,
            }
            options.add_experimental_option("prefs", prefs)
            
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            return True
        except:
            return False
    
    def download_pdf_selenium(self, paper, manual=False):
        """Download PDF using Selenium"""
        if not self.driver:
            if not self.setup_selenium_driver():
                return None
        
        doi = paper.get('doi')
        if not doi:
            return None
        
        url = f"{self.web_base_url}/content/{doi}"
        
        try:
            self.driver.get(url)
            pdf_button = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a.article-dl-pdf-link"))
            )
            
            pdf_button.click()
            self._random_sleep(reason="download")
            
            return "downloaded"
        except:
            return None
    
    def save_metadata(self, paper, pdf_path=None):
        """Save paper metadata"""
        try:
            base_filename = self._get_base_filename(paper)
            txt_filename = f"{base_filename}_metadata.txt"
            txt_path = self.output_dir / txt_filename
            
            txt_content = f"""Title: {paper.get('title', 'N/A')}

Abstract: {paper.get('abstract', 'N/A')}

Date: {paper.get('date', 'N/A')}

DOI: {paper.get('doi', 'N/A')}

"""
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(txt_content)
            print(f"Metadata saved: {txt_filename}")
            return True
        except Exception as e:
            print(f"Metadata save failed: {e}")
            return False
    
    def create_missing_metadata_for_pdfs(self):
        """Check all PDFs and create metadata if missing"""
        print("\n" + "="*60)
        print("Checking for PDFs without metadata...")
        print("="*60)
        
        pdf_files = list(self.output_dir.glob("*.pdf"))
        created_count = 0
        
        for pdf_file in pdf_files:
            pdf_stem = pdf_file.stem
            
            # Check if corresponding metadata exists
            expected_metadata = self.output_dir / f"{pdf_stem}_metadata.txt"
            
            if expected_metadata.exists():
                continue
            
            print(f"\n PDF without metadata: {pdf_file.name}")
            
            # Extract DOI from PDF filename
            doi_part = None
            
            # Try format: 2020.09.30.319780v4.full.pdf
            doi_match = re.search(r'(\d{4}\.\d{2}\.\d{2}\.\d+)', pdf_file.name)
            if doi_match:
                doi_part = doi_match.group(1)
            else:
                # Try format: 008326v4.full.pdf
                doi_match = re.search(r'^(\d{6})v\d+', pdf_file.name)
                if doi_match:
                    doi_part = doi_match.group(1)
            
            if not doi_part:
                print(f"Cannot extract DOI from filename")
                continue
            
            print(f"Extracted DOI: {doi_part}")
            
            # Construct full DOI
            doi = f"10.1101/{doi_part}"
            
            # Try to fetch paper info from bioRxiv API
            try:
                api_url = f"{self.base_url}/details/biorxiv/{doi}"
                response = self.session.get(api_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'collection' in data and len(data['collection']) > 0:
                        paper_info = data['collection'][0]
                        
                        # Create metadata with real info
                        txt_content = f"""Title: {paper_info.get('title', 'N/A')}

Abstract: {paper_info.get('abstract', 'N/A')}

Date: {paper_info.get('date', 'N/A')}

DOI: {doi}
"""
                        
                        metadata_path = self.output_dir / f"{pdf_stem}_metadata.txt"
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            f.write(txt_content)
                        
                        print(f"Created metadata with API data")
                        created_count += 1
                        continue
            except:
                pass
            
            # If API fails, create minimal metadata
            print(f"API failed, creating minimal metadata")
            
            txt_content = f"""Title: Paper {doi_part}

Abstract: N/A (PDF downloaded but metadata not available)

Date: Unknown

DOI: {doi}
"""
            
            metadata_path = self.output_dir / f"{pdf_stem}_metadata.txt"
            try:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    f.write(txt_content)
                print(f"Created minimal metadata")
                created_count += 1
            except Exception as e:
                print(f"Failed to create metadata: {e}")
        
        print("\n" + "="*60)
        print(f"Created {created_count} new metadata files")
        print("="*60)
    
    def rename_downloaded_pdfs(self):
        """Rename all PDFs to match metadata filenames"""
        print("\n" + "="*60)
        print("Renaming PDFs to match metadata...")
        print("="*60)
        
        metadata_files = list(self.output_dir.glob("*_metadata.txt"))
        renamed_count = 0
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doi_match = re.search(r'DOI:\s*(.+)', content)
                    title_match = re.search(r'Title:\s*(.+)', content)
                    
                    if not doi_match or not title_match:
                        continue
                    
                    doi = doi_match.group(1).strip()
                    title = title_match.group(1).strip()
                
                paper = {'doi': doi, 'title': title}
                base_filename = self._get_base_filename(paper)
                expected_pdf = self.output_dir / f"{base_filename}.pdf"
                
                if expected_pdf.exists():
                    continue
                
                # Find the original PDF
                doi_parts = doi.split('/')[-1]
                possible_patterns = [
                    f"*{doi_parts}*.pdf",
                ]
                
                original_pdf = None
                for pattern in possible_patterns:
                    matches = list(self.output_dir.glob(pattern))
                    matches = [m for m in matches if m != expected_pdf and '_metadata' not in m.name]
                    if matches:
                        original_pdf = matches[0]
                        break
                
                if original_pdf and original_pdf.exists():
                    original_pdf.rename(expected_pdf)
                    print(f"Renamed: {original_pdf.name} -> {expected_pdf.name}")
                    renamed_count += 1
            except:
                continue
        
        print(f"\nSuccessfully renamed: {renamed_count} files")
    
    def download_papers(self, query="uncertainty prediction ML", days_back=200,
                       max_results=20, method="auto", manual_mode=False, start_year=None):
        """Main function to search and download papers"""
        print(f"Starting bioRxiv paper download: {query}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print("-" * 60)
        
        papers = []
        if method in ['api', 'auto']:
            papers = self.search_papers_api(query, days_back, max_results, start_year=start_year)
        
        if not papers:
            print("No papers found")
            return
        
        successful_downloads = 0
        failed_downloads = 0
        
        for i, paper in enumerate(papers, 1):
            print(f"\n{'='*60}")
            print(f"📑 Processing paper {i}/{len(papers)}:")
            print(f"Title: {paper.get('title', 'Unknown')[:60]}...")
            
            # Save metadata first
            self.save_metadata(paper, None)
            
            # Try to download PDF
            pdf_path = self.download_pdf_direct(paper)
            
            if not pdf_path and SELENIUM_AVAILABLE:
                print("Trying Selenium...")
                pdf_path = self.download_pdf_selenium(paper, manual=manual_mode)
            
            if pdf_path:
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            if i < len(papers):
                self._random_sleep(reason="rate_limit")
        
        if self.driver:
            self.driver.quit()
        
        # Rename all PDFs to match metadata
        self.rename_downloaded_pdfs()
        
        # Check for PDFs without metadata and create them
        self.create_missing_metadata_for_pdfs()
        
        print("\n" + "=" * 60)
        print(f"Final Summary:")
        print(f"Total papers: {len(papers)}")
        print(f"Successful: {successful_downloads}")
        print(f"Failed: {failed_downloads}")
        print("=" * 60)


def main():
    """Main execution"""
    query = "uncertainty prediction machine learning"
    output_dir = "uncertainty_papers"
    start_year = 2020
    max_results = 100
    
    downloader = EnhancedBioRxivDownloader(output_dir)
    
    downloader.download_papers(
        query=query,
        max_results=max_results,
        method="api",
        manual_mode=False,
        start_year=start_year
    )


if __name__ == "__main__":
    main()