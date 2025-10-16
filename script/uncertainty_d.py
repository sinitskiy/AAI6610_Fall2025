#!/usr/bin/env python3

import requests
import json
import os
import time
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
        
        # Setup session for API calls
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Setup selenium driver if available
        self.driver = None
        
    def search_papers_api(self, query="uncertainty prediction ML", days_back=200, max_results=50):
        """
        Search for papers using bioRxiv API (Method 1 - More reliable)
        """
        print(f"API Search: '{query}' (last {days_back} days)")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Search using bioRxiv API
        search_url = f"{self.base_url}/details/biorxiv/{start_date_str}/{end_date_str}"
        
        try:
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'collection' not in data:
                print("📭  No papers found in date range")
                return []
            
            papers = data['collection']
            print(f"📑 Found {len(papers)} papers in date range")
            
            # Flexible matching
            filtered_papers = []
            query_terms = [term.lower() for term in query.split()]
            
            for paper in papers:
                title = paper.get('title', '').lower()
                abstract = paper.get('abstract', '').lower()
                combined_text = f"{title} {abstract}"
                
                # Count how many query terms match
                matches = sum(1 for term in query_terms if term in combined_text)
                match_ratio = matches / len(query_terms) if query_terms else 0
                
                # Accept if at least 50% of terms match
                if match_ratio >= 0.5:
                    paper['_match_score'] = match_ratio
                    filtered_papers.append(paper)
            
            # Sort by match score (highest first)
            filtered_papers.sort(key=lambda p: p.get('_match_score', 0), reverse=True)
            filtered_papers = filtered_papers[:max_results]
            
            print(f"Found {len(filtered_papers)} papers matching '{query}'")
            return filtered_papers[:max_results]
            
        except requests.RequestException as e:
            print(f"API search error: {e}")
            return []

    def search_papers_web(self, query="uncertainty prediction ML", max_results=50):
        """
        Search for papers using web scraping with better anti-bot measures
        """
        print(f"Web Search: '{query}'")
        
        # Use different search approaches
        search_methods = [
            # Method 1: Direct search
            f"{self.web_base_url}/search/{quote(query)}",
            # Method 2: Split query approach
            f"{self.web_base_url}/search/{quote(query.split()[0])}",
            # Method 3: Alternative search format
            f"{self.web_base_url}/search?query={quote(query)}"
        ]
        
        # Enhanced headers to avoid 403
        enhanced_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Referer': 'https://www.biorxiv.org/',
            'Upgrade-Insecure-Requests': '1',
        }
        
        for search_url in search_methods:
            print(f"Trying search URL: {search_url}")
            
            try:
                # Add delay to display human-like
                time.sleep(3)
                
                response = self.session.get(search_url, headers=enhanced_headers, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Try multiple selectors for DOIs
                    doi_selectors = [
                        "span.highwire-citation-doi",
                        ".citation-doi",
                        "a[href*='doi.org']",
                        ".doi"
                    ]
                    
                    articles = []
                    for selector in doi_selectors:
                        articles = soup.select(selector)
                        if articles:
                            break
                    
                    if not articles:
                        # Try to find article links instead
                        article_links = soup.select("a[href*='/content/']")
                        dois = []
                        for link in article_links[:max_results]:
                            href = link.get('href', '')
                            # Extract DOI pattern from URL
                            match = re.search(r'/content/(?:early/\d{4}/\d{2}/\d{2}/)?([^v?\s]+)', href)
                            if match:
                                dois.append(match.group(1))
                    else:
                        dois = []
                        for article in articles[:max_results]:
                            doi_text = article.text.strip()
                            if "https://doi.org/" in doi_text:
                                doi = doi_text.replace("https://doi.org/", "")
                            elif "doi.org/" in doi_text:
                                doi = doi_text.split("doi.org/")[-1]
                            else:
                                doi = doi_text
                            
                            if doi and '10.' in doi:  # Basic DOI validation
                                dois.append(doi)
                    
                    if dois:
                        print(f"Web search found {len(dois)} DOIs")
                        
                        # Convert DOIs to paper objects
                        papers = []
                        for doi in dois:
                            # Try to get title
                            title = f"Paper {doi.split('/')[-1]}"
                            try:
                                title_element = soup.find('h1', class_='highwire-cite-title')
                                if title_element:
                                    title = title_element.get_text(strip=True)
                            except:
                                pass
                            
                            papers.append({
                                'doi': doi,
                                'title': title,
                                'date': datetime.now().strftime("%Y-%m-%d"),
                                'abstract': 'N/A (obtained from web search)',
                                '_source': 'web_search'
                            })
                        
                        return papers
                
                elif response.status_code == 403:
                    print(f"403 Forbidden error, trying next method")
                    continue
                else:
                    print(f"HTTP {response.status_code} error, trying next method")
                    continue
                    
            except requests.RequestException as e:
                print(f"Request error: {e}")
                continue
        
        print("No papers found via web search")
        return []

    def download_pdf_direct(self, paper):
        """
        Direct PDF download (Method 1)
        """
        doi = paper.get('doi')
        if not doi:
            print("No DOI found")
            return None
            
        # Try different PDF URL formats
        pdf_urls = [
            f"{self.web_base_url}/content/{doi}v{paper.get('version', '1')}.full.pdf",
            f"{self.web_base_url}/content/{doi}.full.pdf",
            f"{self.web_base_url}/content/early/{datetime.now().strftime('%Y/%m/%d')}/{doi}.full.pdf"
        ]
        
        # Create safe filename
        safe_title = self._create_safe_filename(paper.get('title', 'unknown'))
        filename = f"{safe_title}_{doi.split('/')[-1]}.pdf"
        filepath = self.output_dir / filename
        
        for pdf_url in pdf_urls:
            try:
                print(f"Attempting download: {paper.get('title', 'Unknown title')[:50]}...")
                response = self.session.get(pdf_url, timeout=60)
                
                if response.status_code == 200 and 'pdf' in response.headers.get('content-type', '').lower():
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"Download successful: {filename}")
                    return filepath
                    
            except requests.RequestException as e:
                continue
        
        print(f"Download failed: {filename}")
        return None

    def setup_selenium_driver(self):
        """
        Setup Selenium driver for manual download assistance
        """
        if not SELENIUM_AVAILABLE:
            print("False to use Selenium.")
            return False
            
        try:
            options = webdriver.ChromeOptions()
            prefs = {
                "download.default_directory": str(self.output_dir.absolute()),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "plugins.always_open_pdf_externally": True,
            }
            options.add_experimental_option("prefs", prefs)
            
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), 
                options=options
            )
            return True
        except Exception as e:
            print(f"Selenium setting failed: {e}")
            return False

    def download_pdf_selenium(self, paper, manual=False):
        """
        Download PDF using Selenium (Method 2)
        """
        if not self.driver:
            if not self.setup_selenium_driver():
                return None
                
        doi = paper.get('doi')
        if not doi:
            return None
            
        url = f"{self.web_base_url}/content/{doi}"
        print(f"Opening paper page: {url}")
        
        try:
            self.driver.get(url)
            
            # Wait for PDF download button
            pdf_button = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a.article-dl-pdf-link"))
            )
            
            self.driver.execute_script("arguments[0].scrollIntoView();", pdf_button)
            
            if manual:
                print("Found PDF download button, please click manually to download.")
                time.sleep(25)  # Wait for manual click
            else:
                print("Auto-clicking download button...")
                pdf_button.click()
                time.sleep(8)  # Wait for download to start
            
            return "downloaded"
            
        except Exception as e:
            print(f"Selenium download failed: {e}")
            return None

    def _create_safe_filename(self, title):
        """
        Create safe filename from title
        """
        safe_title = re.sub(r'[^\w\s-]', '', title)
        safe_title = re.sub(r'[-\s]+', '-', safe_title)[:100]
        return safe_title.strip('-')

    def save_metadata(self, paper, pdf_path=None):
        """
        Save paper metadata as text file
        """
        doi = paper.get('doi', 'unknown')
        safe_title = self._create_safe_filename(paper.get('title', 'unknown'))
        
        txt_content_filename = f"{safe_title}_{doi.split('/')[-1]}_metadata.txt"
        txt_content_path = self.output_dir / txt_content_filename
        
        # Format metadata
        txt_content = f"""Title: {paper.get('title', 'N/A')}

Abstract: {paper.get('abstract', 'N/A')}

Date: {paper.get('date', 'N/A')}

DOI: {paper.get('doi', 'N/A')}
"""
        
        try:
            with open(txt_content_path, 'w', encoding='utf-8') as f:
                f.write(txt_content)
            print(f"Txt content saved: {txt_content_filename}")
            return txt_content_path
        except Exception as e:
            print(f"Txt content failed: {e}")
            return None

    def download_papers(self, query="uncertainty prediction ML", days_back=200, 
                       max_results=20, method="auto", manual_mode=False):
        """
        Main function to search and download papers
        Methods: 'api', 'web', 'auto' (try api first, then web)
        """
        print(f"Starting bioRxiv paper download: {query}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print("-" * 60)
        
        # Search for papers
        papers = []
        if method in ['api', 'auto']:
            papers = self.search_papers_api(query, days_back, max_results)
            
        if not papers and method in ['web', 'auto']:
            print("API search returned no results, trying web scraping....")
            papers = self.search_papers_web(query, max_results)
        
        if not papers:
            print("Not found any papers matching the query.")
            return
        
        # Download papers and save metadata
        successful_downloads = 0
        failed_downloads = 0
        
        for i, paper in enumerate(papers, 1):
            print(f"\n Processing paper {i}/{len(papers)}:")
            print(f"Title: {paper.get('title', 'Unknown')[:60]}...")
            
            # Try direct download first
            pdf_path = self.download_pdf_direct(paper)
            
            # If direct download fails, try Selenium
            if not pdf_path and SELENIUM_AVAILABLE:
                print("Download failed, trying Selenium")
                pdf_path = self.download_pdf_selenium(paper, manual=manual_mode)
            
            if pdf_path:
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            # Always save metadata
            self.save_metadata(paper, pdf_path)
            
            # Rate limiting
            time.sleep(2)
        
        # Cleanup
        if self.driver:
            self.driver.quit()
        
        print("\n" + "=" * 60)
        print(f"Total papers processed: {len(papers)}")
        print(f"Successful downloads: {successful_downloads}")
        print(f"Failed downloads: {failed_downloads}")
        print(f"Output directory: {self.output_dir.absolute()}")

def main():
    """
    Main execution function
    """
    # Setting configurations
    query = "uncertainty prediction"  # Simplified keywords to improve matching rate
    output_dir = "uncertainty_papers"
    days_back = 200
    max_results = 20
       
    # Create downloader instance
    downloader = EnhancedBioRxivDownloader(output_dir)
    
    # Download papers
    # method options: 'api', 'web', 'auto'
    # manual_mode: True for manual clicking, False for automatic
    downloader.download_papers(
        query=query,
        days_back=days_back,
        max_results=max_results,
        method="auto",        # Try API first, then web scraping
        manual_mode=False     # Set to True if automatic clicking fails
    )

if __name__ == "__main__":
    main()
