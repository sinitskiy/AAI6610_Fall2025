#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LinkedIn Scraper v2.0 - Free Version
- Uses DuckDuckGo search (free, no API key)
- Optional Google Custom Search (if API key provided)
- Better rate limiting to avoid bans
- Cleaner code

WARNING: LinkedIn actively blocks scrapers. This may not work reliably.
Consider using Reddit/arXiv instead for ML research discussions.
"""

import os
import re
import json
import time
import random
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Set, Optional

import requests
from bs4 import BeautifulSoup

# Playwright for browser automation
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("WARNING: playwright not installed. Run: pip install playwright && playwright install")

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

# Paths
ENV_PATHS = [PIPELINE_DIR / ".env", PROJECT_ROOT / ".env"]
COOKIES_FILE = SCRIPT_DIR / "linkedin_cookies.json"
OUTPUT_DIR = SCRAPERS_DIR / "outputs" / "linkedin_posts"

# ============================================================================
# Load Environment Variables
# ============================================================================
def load_env():
    for env_path in ENV_PATHS:
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip())
            return

load_env()

# ============================================================================
# Configuration
# ============================================================================
# Google API (optional - has free tier of 100 queries/day)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
GOOGLE_SEARCH_ENGINE_ID = os.environ.get('GOOGLE_SEARCH_ENGINE_ID', '')

# Search keywords for ML uncertainty
KEYWORDS = [
    "uncertainty quantification machine learning",
    "bayesian deep learning uncertainty",
    "confidence calibration neural networks",
    "epistemic aleatoric uncertainty",
    "probabilistic prediction deep learning",
    "monte carlo dropout uncertainty",
    "ensemble uncertainty estimation",
]

# Limits
MAX_POSTS_PER_KEYWORD = 30
MIN_DELAY = 5  # Increased to avoid detection
MAX_DELAY = 12

# ============================================================================
# URL Search Functions
# ============================================================================
def search_duckduckgo(query: str, max_results: int = 30) -> List[str]:
    """
    Search DuckDuckGo for LinkedIn posts - FREE, no API key needed
    """
    logger.info(f"  DuckDuckGo: {query[:40]}...")
    
    urls = []
    search_query = f"{query} site:linkedin.com/posts"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        # DuckDuckGo HTML search
        search_url = "https://html.duckduckgo.com/html/"
        params = {'q': search_query}
        
        response = requests.post(search_url, data=params, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract result links
        for result in soup.select('.result__a'):
            href = result.get('href', '')
            
            # DuckDuckGo wraps URLs, need to extract actual URL
            if 'uddg=' in href:
                # Extract the actual URL from DuckDuckGo's redirect
                import urllib.parse
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                if 'uddg' in parsed:
                    actual_url = parsed['uddg'][0]
                    if 'linkedin.com/posts' in actual_url:
                        urls.append(actual_url)
            elif 'linkedin.com/posts' in href:
                urls.append(href)
            
            if len(urls) >= max_results:
                break
        
        logger.info(f"    Found: {len(urls)} URLs")
        
    except Exception as e:
        logger.warning(f"    DuckDuckGo error: {e}")
    
    time.sleep(random.uniform(2, 4))
    return urls[:max_results]

def search_google(query: str, max_results: int = 30) -> List[str]:
    """
    Search Google Custom Search API - requires API key (100 free queries/day)
    """
    if not GOOGLE_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        return []
    
    logger.info(f"  Google API: {query[:40]}...")
    
    urls = []
    search_query = f"{query} site:linkedin.com/posts"
    
    try:
        for start in range(1, max_results + 1, 10):
            params = {
                'key': GOOGLE_API_KEY,
                'cx': GOOGLE_SEARCH_ENGINE_ID,
                'q': search_query,
                'num': min(10, max_results - len(urls)),
                'start': start,
            }
            
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
                timeout=15
            )
            
            if response.status_code != 200:
                logger.warning(f"    Google API error: {response.status_code}")
                break
            
            items = response.json().get('items', [])
            
            for item in items:
                link = item.get('link', '')
                if 'linkedin.com/posts' in link:
                    urls.append(link)
            
            if len(urls) >= max_results:
                break
            
            time.sleep(random.uniform(1, 2))
        
        logger.info(f"    Found: {len(urls)} URLs")
        
    except Exception as e:
        logger.warning(f"    Google API error: {e}")
    
    return urls[:max_results]

def search_linkedin_urls(query: str, max_results: int = 30) -> List[str]:
    """
    Search for LinkedIn post URLs using available methods
    """
    urls = []
    
    # Try Google first (if API key available)
    if GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID:
        urls = search_google(query, max_results)
    
    # Fallback to DuckDuckGo (free)
    if len(urls) < max_results:
        ddg_urls = search_duckduckgo(query, max_results - len(urls))
        urls.extend(ddg_urls)
    
    # Deduplicate
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls[:max_results]

# ============================================================================
# LinkedIn Scraper
# ============================================================================
class LinkedInScraper:
    """LinkedIn post scraper using Playwright"""
    
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seen_urls: Set[str] = set()
        self.saved_count = 0
    
    def url_to_filename(self, url: str) -> str:
        """Generate unique filename from URL"""
        hash_id = hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
        return f"linkedin_{hash_id}.txt"
    
    def scrape_post(self, url: str) -> bool:
        """Scrape a single LinkedIn post"""
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Playwright not available!")
            return False
        
        filename = self.output_dir / self.url_to_filename(url)
        
        if filename.exists():
            logger.debug(f"  Already scraped: {url[:50]}...")
            return True
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                
                # Load cookies if available
                if COOKIES_FILE.exists():
                    try:
                        with open(COOKIES_FILE, 'r', encoding='utf-8') as f:
                            cookies = json.load(f)
                            context.add_cookies(cookies)
                    except:
                        pass
                
                page = context.new_page()
                
                try:
                    page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    page.wait_for_timeout(random.randint(2000, 4000))
                except Exception as e:
                    logger.debug(f"  Page load error: {e}")
                    browser.close()
                    return False
                
                # Extract content
                content = ""
                author = "Unknown"
                date_text = datetime.now().strftime("%Y-%m-%d")
                
                # Try different selectors for post content
                selectors = [
                    'div.feed-shared-update-v2__description',
                    'div.feed-shared-text',
                    'div.update-components-text',
                    'article',
                    'main',
                ]
                
                for selector in selectors:
                    try:
                        element = page.query_selector(selector)
                        if element:
                            content = element.inner_text()
                            if len(content) > 50:
                                break
                    except:
                        continue
                
                if not content or len(content) < 50:
                    # Fallback to body
                    try:
                        content = page.inner_text('body')[:3000]
                    except:
                        content = ""
                
                # Try to get author
                author_selectors = [
                    'span.feed-shared-actor__name',
                    'a.feed-shared-actor__container-link',
                    '.update-components-actor__name',
                ]
                
                for selector in author_selectors:
                    try:
                        element = page.query_selector(selector)
                        if element:
                            author = element.inner_text().strip()
                            if author:
                                break
                    except:
                        continue
                
                # Try to get date
                date_selectors = [
                    'span.feed-shared-actor__sub-description',
                    'time',
                    '.update-components-actor__sub-description',
                ]
                
                for selector in date_selectors:
                    try:
                        element = page.query_selector(selector)
                        if element:
                            date_text = element.inner_text().strip()[:50]
                            break
                    except:
                        continue
                
                browser.close()
                
                # Skip if no meaningful content
                if not content or len(content.strip()) < 100:
                    logger.debug(f"  No content: {url[:50]}...")
                    return False
                
                # Save to file
                lines = [
                    f"Source: LinkedIn",
                    f"URL: {url}",
                    f"Author: {author}",
                    f"Date: {date_text}",
                    "",
                    "=" * 80,
                    "",
                    "Content:",
                    content.strip(),
                ]
                
                filename.write_text('\n'.join(lines), encoding='utf-8')
                self.saved_count += 1
                logger.info(f"  Saved: {filename.name}")
                
                return True
                
        except Exception as e:
            logger.debug(f"  Scrape error: {e}")
            return False
        
        finally:
            # Random delay to avoid detection
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    
    def run(self) -> int:
        """Run the scraper"""
        print(f"\n{'#'*70}")
        print("# LinkedIn Scraper v2.0 (Free Version)")
        print(f"{'#'*70}")
        print(f"Keywords: {len(KEYWORDS)}")
        print(f"Max posts per keyword: {MAX_POSTS_PER_KEYWORD}")
        print(f"Output: {self.output_dir}")
        
        if GOOGLE_API_KEY:
            print("Search: Google API + DuckDuckGo")
        else:
            print("Search: DuckDuckGo (free)")
        
        print(f"{'#'*70}\n")
        
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Playwright not installed!")
            logger.info("Run: pip install playwright && playwright install chromium")
            return 0
        
        # Collect all URLs
        all_urls: Set[str] = set()
        
        for keyword in KEYWORDS:
            logger.info(f"\nSearching: {keyword}")
            
            urls = search_linkedin_urls(keyword, MAX_POSTS_PER_KEYWORD)
            all_urls.update(urls)
            
            logger.info(f"  Total unique URLs: {len(all_urls)}")
            time.sleep(random.uniform(2, 4))
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Total URLs to scrape: {len(all_urls)}")
        logger.info(f"{'='*50}\n")
        
        # Scrape posts
        url_list = list(all_urls)
        
        for idx, url in enumerate(url_list, 1):
            logger.info(f"[{idx}/{len(url_list)}] {url[:60]}...")
            
            try:
                self.scrape_post(url)
            except Exception as e:
                logger.warning(f"  Error: {e}")
            
            # Progress update
            if idx % 10 == 0:
                logger.info(f"  Progress: {idx}/{len(url_list)}, Saved: {self.saved_count}")
        
        # Summary
        print(f"\n{'#'*70}")
        print("# SCRAPING COMPLETE")
        print(f"{'#'*70}")
        print(f"URLs processed: {len(url_list)}")
        print(f"Posts saved: {self.saved_count}")
        print(f"Output: {self.output_dir}")
        print(f"Files: {len(list(self.output_dir.glob('*.txt')))}")
        print(f"{'#'*70}\n")
        
        return self.saved_count

# ============================================================================
# Cookie Helper
# ============================================================================
def save_linkedin_cookies():
    """
    Helper to save LinkedIn cookies for authenticated scraping.
    Run this once after logging into LinkedIn manually.
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("Playwright not installed!")
        return
    
    print("\n" + "="*60)
    print("LinkedIn Cookie Saver")
    print("="*60)
    print("\n1. A browser window will open")
    print("2. Log into your LinkedIn account")
    print("3. After logging in, press Enter in this terminal")
    print("4. Cookies will be saved for future use")
    print("\n" + "="*60)
    
    input("\nPress Enter to open browser...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        page.goto("https://www.linkedin.com/login")
        
        input("\nLog in to LinkedIn, then press Enter here...")
        
        cookies = context.cookies()
        
        with open(COOKIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(cookies, f, indent=2)
        
        print(f"\nCookies saved to: {COOKIES_FILE}")
        browser.close()

# ============================================================================
# Main
# ============================================================================
def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--save-cookies':
        save_linkedin_cookies()
    else:
        scraper = LinkedInScraper()
        scraper.run()

if __name__ == "__main__":
    main()