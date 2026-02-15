#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reddit + StackExchange Scraper v2.0
- Fixed keywords for ML uncertainty research
- Reads settings from config.yaml
- Better error handling and rate limiting
"""

import os
import re
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import praw
import requests
from bs4 import BeautifulSoup
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
CONFIG_PATHS = [PIPELINE_DIR / "config.yaml", PROJECT_ROOT / "config.yaml"]
ENV_PATHS = [PIPELINE_DIR / ".env", PROJECT_ROOT / ".env"]
OUTPUT_DIR = SCRAPERS_DIR / "outputs" / "reddit_posts"

# ============================================================================
# Load Environment Variables
# ============================================================================
def load_env():
    """Load .env file"""
    for env_path in ENV_PATHS:
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip())
            logger.info(f"Loaded env from {env_path.name}")
            return True
    logger.warning("No .env file found")
    return False

load_env()

# ============================================================================
# Load Configuration
# ============================================================================
def load_config() -> dict:
    """Load configuration from config.yaml"""
    for path in CONFIG_PATHS:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except:
                pass
    return {}

CONFIG = load_config()
SCRAPER_CONFIG = CONFIG.get('scrapers', {}).get('reddit', {})

# ============================================================================
# Configuration
# ============================================================================
# Reddit API credentials
REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID', '')
REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET', '')
REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT', 'AAI6610-Pipeline/2.0')

# Subreddits to search (ML/AI focused)
SUBREDDITS = SCRAPER_CONFIG.get('subreddits', [
    'MachineLearning',
    'deeplearning',
    'artificial',
    'statistics',
    'datascience',
    'learnmachinelearning',
])

# Search keywords - FIXED for ML uncertainty research
SEARCH_KEYWORDS = [
    # Core uncertainty topics
    "uncertainty quantification",
    "uncertainty estimation",
    "predictive uncertainty",
    "model uncertainty",
    "confidence calibration",
    
    # Methods
    "bayesian neural network",
    "monte carlo dropout",
    "deep ensemble",
    "conformal prediction",
    
    # Related concepts
    "epistemic uncertainty",
    "aleatoric uncertainty",
    "out of distribution detection",
    "calibration error",
    "prediction intervals",
    
    # Applications
    "uncertainty in deep learning",
    "reliable machine learning",
    "trustworthy AI",
]

# StackExchange sites
STACKEXCHANGE_SITES = [
    'datascience.stackexchange.com',
    'stats.stackexchange.com',
    'ai.stackexchange.com',
]

# Limits
MAX_POSTS_PER_SEARCH = SCRAPER_CONFIG.get('max_posts', 500) // len(SEARCH_KEYWORDS)
MAX_COMMENTS = 15
RATE_LIMIT_DELAY = 2.0

# ============================================================================
# Utility Functions
# ============================================================================
def sanitize_filename(text: str, max_length: int = 120) -> str:
    """Clean text for use as filename"""
    # Remove invalid characters
    clean = re.sub(r'[\\/*?:"<>|\n\r\t]', '_', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    clean = re.sub(r'_+', '_', clean)
    return clean[:max_length]

def save_post(title: str, content: str, comments: List[str], 
              url: str, source: str, date: str = None,
              author: str = None, score: int = None):
    """Save post to text file"""
    if not title.strip():
        title = "untitled"
    
    filename = sanitize_filename(title) + ".txt"
    filepath = OUTPUT_DIR / filename
    
    # Handle duplicate filenames
    counter = 1
    while filepath.exists():
        filename = f"{sanitize_filename(title)}_{counter}.txt"
        filepath = OUTPUT_DIR / filename
        counter += 1
    
    lines = [
        f"Source: {source}",
        f"URL: {url}",
    ]
    
    if date:
        lines.append(f"Date: {date}")
    if author:
        lines.append(f"Author: {author}")
    if score is not None:
        lines.append(f"Score: {score}")
    
    lines.extend([
        f"Title: {title}",
        "",
        "Content:",
        content or "(no content)",
        "",
        "Comments:",
    ])
    
    for c in comments[:MAX_COMMENTS]:
        lines.append(f"- {c[:500]}")  # Limit comment length
    
    filepath.write_text('\n'.join(lines), encoding='utf-8')
    logger.debug(f"Saved: {filename}")
    
    return filepath

# ============================================================================
# Reddit Scraper
# ============================================================================
class RedditScraper:
    """Scraper for Reddit posts"""
    
    def __init__(self):
        if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
            raise ValueError("Reddit API credentials not set in .env file")
        
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        self.saved_count = 0
        self.seen_ids = set()
    
    def search_subreddit(self, subreddit_name: str, keyword: str, limit: int = 50) -> int:
        """Search a subreddit for posts matching keyword"""
        saved = 0
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Search with different sort options
            for sort in ['relevance', 'new', 'top']:
                try:
                    results = subreddit.search(keyword, sort=sort, limit=limit, time_filter='year')
                    
                    for post in results:
                        if post.id in self.seen_ids:
                            continue
                        
                        self.seen_ids.add(post.id)
                        
                        # Extract post data
                        title = post.title
                        content = post.selftext or ""
                        url = f"https://reddit.com{post.permalink}"
                        author = str(post.author) if post.author else "deleted"
                        score = post.score
                        
                        try:
                            date = datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            date = None
                        
                        # Get comments
                        comments = []
                        try:
                            post.comments.replace_more(limit=0)
                            for comment in post.comments[:MAX_COMMENTS]:
                                if hasattr(comment, 'body'):
                                    comments.append(comment.body.strip())
                        except:
                            pass
                        
                        # Save post
                        save_post(
                            title=title,
                            content=content,
                            comments=comments,
                            url=url,
                            source=f"Reddit/r/{subreddit_name}",
                            date=date,
                            author=author,
                            score=score
                        )
                        
                        saved += 1
                        time.sleep(RATE_LIMIT_DELAY)
                        
                except Exception as e:
                    logger.debug(f"Search error ({sort}): {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Subreddit error (r/{subreddit_name}): {e}")
        
        return saved
    
    def scrape(self) -> int:
        """Main scraping function"""
        logger.info("Starting Reddit scraper...")
        logger.info(f"  Subreddits: {len(SUBREDDITS)}")
        logger.info(f"  Keywords: {len(SEARCH_KEYWORDS)}")
        
        total_saved = 0
        
        for subreddit in SUBREDDITS:
            logger.info(f"\nSearching r/{subreddit}...")
            
            for keyword in SEARCH_KEYWORDS:
                logger.info(f"  Keyword: {keyword[:40]}...")
                
                saved = self.search_subreddit(
                    subreddit, keyword, 
                    limit=MAX_POSTS_PER_SEARCH
                )
                
                total_saved += saved
                
                if saved > 0:
                    logger.info(f"    Found: {saved} posts (total: {total_saved})")
                
                time.sleep(RATE_LIMIT_DELAY)
        
        logger.info(f"\nReddit scraping complete: {total_saved} posts")
        return total_saved

# ============================================================================
# StackExchange Scraper
# ============================================================================
class StackExchangeScraper:
    """Scraper for StackExchange sites"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.seen_urls = set()
    
    def search_site(self, site: str, keyword: str, max_pages: int = 3) -> int:
        """Search a StackExchange site"""
        saved = 0
        
        for page in range(1, max_pages + 1):
            search_url = f"https://{site}/search"
            params = {
                'page': page,
                'tab': 'newest',
                'q': keyword,
            }
            
            try:
                resp = self.session.get(search_url, params=params, timeout=15)
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                questions = soup.select('a.question-hyperlink')
                
                if not questions:
                    break
                
                for q in questions:
                    q_url = f"https://{site}{q.get('href')}"
                    
                    if q_url in self.seen_urls:
                        continue
                    
                    self.seen_urls.add(q_url)
                    
                    try:
                        # Fetch question page
                        q_resp = self.session.get(q_url, timeout=15)
                        q_soup = BeautifulSoup(q_resp.text, 'html.parser')
                        
                        # Extract data
                        title = q.text.strip()
                        
                        content_elem = q_soup.select_one('.js-post-body')
                        content = content_elem.get_text('\n').strip() if content_elem else ""
                        
                        # Get answers as "comments"
                        answers = []
                        for ans in q_soup.select('.answer .js-post-body')[:5]:
                            answers.append(ans.get_text('\n').strip()[:500])
                        
                        # Get date
                        date = None
                        time_elem = q_soup.select_one('time.relativetime')
                        if time_elem and time_elem.get('datetime'):
                            try:
                                dt = datetime.fromisoformat(time_elem['datetime'].replace('Z', '+00:00'))
                                date = dt.strftime("%Y-%m-%d %H:%M:%S")
                            except:
                                pass
                        
                        # Save
                        save_post(
                            title=title,
                            content=content,
                            comments=answers,
                            url=q_url,
                            source=f"StackExchange/{site}",
                            date=date
                        )
                        
                        saved += 1
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.debug(f"Question fetch error: {e}")
                        continue
                
                time.sleep(2)
                
            except Exception as e:
                logger.debug(f"Search page error: {e}")
                continue
        
        return saved
    
    def scrape(self) -> int:
        """Main scraping function"""
        logger.info("\nStarting StackExchange scraper...")
        logger.info(f"  Sites: {len(STACKEXCHANGE_SITES)}")
        
        total_saved = 0
        
        # Use subset of keywords for StackExchange
        keywords = SEARCH_KEYWORDS[:8]
        
        for site in STACKEXCHANGE_SITES:
            logger.info(f"\nSearching {site}...")
            
            for keyword in keywords:
                logger.info(f"  Keyword: {keyword[:30]}...")
                
                saved = self.search_site(site, keyword, max_pages=3)
                total_saved += saved
                
                if saved > 0:
                    logger.info(f"    Found: {saved} questions")
                
                time.sleep(2)
        
        logger.info(f"\nStackExchange scraping complete: {total_saved} questions")
        return total_saved

# ============================================================================
# Main
# ============================================================================
def main():
    print(f"\n{'#'*70}")
    print(f"# Reddit + StackExchange Scraper v2.0")
    print(f"# ML Uncertainty Research")
    print(f"{'#'*70}\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check credentials
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        logger.error("Reddit API credentials not found!")
        logger.info("Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file")
        logger.info("Get credentials at: https://www.reddit.com/prefs/apps")
        return
    
    logger.info(f"Output: {OUTPUT_DIR}")
    
    total = 0
    
    # Reddit
    try:
        reddit_scraper = RedditScraper()
        total += reddit_scraper.scrape()
    except Exception as e:
        logger.error(f"Reddit scraper failed: {e}")
    
    # StackExchange
    try:
        se_scraper = StackExchangeScraper()
        total += se_scraper.scrape()
    except Exception as e:
        logger.error(f"StackExchange scraper failed: {e}")
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"# SCRAPING COMPLETE")
    print(f"{'#'*70}")
    print(f"\nTotal posts saved: {total}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Files: {len(list(OUTPUT_DIR.glob('*.txt')))}")
    print()

if __name__ == "__main__":
    main()