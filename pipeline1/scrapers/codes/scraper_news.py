#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
News Scraper v2.0 - Multi-source news collection
- Reads settings from config.yaml
- Fixed security (no hardcoded API keys)
- Better error handling
"""

import os
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

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
OUTPUT_DIR = SCRAPERS_DIR / "outputs" / "news_articles"

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
            return
    logger.warning("No .env file found")

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
SCRAPER_CONFIG = CONFIG.get('scrapers', {}).get('news', {})

# ============================================================================
# Configuration
# ============================================================================
# API Keys (from environment only - no hardcoding!)
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')

# Target articles
TARGET_ARTICLES = SCRAPER_CONFIG.get('target_articles', 800)

# Search topics - ML Uncertainty focused
TOPICS = [
    "uncertainty quantification machine learning",
    "bayesian deep learning",
    "predictive uncertainty AI",
    "neural network confidence calibration",
    "probabilistic machine learning",
    "monte carlo dropout deep learning",
    "ensemble uncertainty estimation",
    "out of distribution detection",
    "conformal prediction machine learning",
    "trustworthy AI uncertainty",
]

# ============================================================================
# News Scraper Class
# ============================================================================
class NewsScraper:
    """Multi-source news scraper"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.seen_urls = set()
        self.articles = []
        self.stats = {
            'google_news': 0,
            'gdelt': 0,
            'newsapi': 0,
            'tech_blogs': 0,
        }
    
    def scrape_google_news_rss(self, topic: str, max_articles: int = 100) -> List[Dict]:
        """Scrape Google News RSS - Free, no API key needed"""
        logger.info(f"  Google News RSS: {topic[:40]}...")
        
        articles = []
        
        # Search variations for more results
        search_terms = [
            topic,
            f"{topic} research",
            f"{topic} news",
            f"{topic} 2024",
        ]
        
        for term in search_terms:
            if len(articles) >= max_articles:
                break
            
            query = term.replace(' ', '+')
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"
            
            try:
                response = self.session.get(rss_url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                for item in items:
                    url = item.find('link')
                    url = url.text if url else ''
                    
                    if not url or url in self.seen_urls:
                        continue
                    
                    self.seen_urls.add(url)
                    
                    title = item.find('title')
                    desc = item.find('description')
                    pub_date = item.find('pubDate')
                    
                    articles.append({
                        'title': title.text if title else '',
                        'description': desc.text if desc else '',
                        'url': url,
                        'source': 'Google News',
                        'published_at': pub_date.text if pub_date else '',
                        'content': '',
                    })
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.debug(f"    RSS error: {e}")
        
        self.stats['google_news'] += len(articles)
        logger.info(f"    Found: {len(articles)} articles")
        return articles[:max_articles]
    
    def scrape_gdelt(self, topic: str, max_articles: int = 80) -> List[Dict]:
        """Scrape GDELT - Free, no API key needed"""
        logger.info(f"  GDELT: {topic[:40]}...")
        
        articles = []
        
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            'query': topic,
            'mode': 'artlist',
            'maxrecords': max_articles,
            'format': 'json',
            'sort': 'datedesc',
        }
        
        try:
            response = self.session.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            for article in data.get('articles', []):
                art_url = article.get('url', '')
                
                if not art_url or art_url in self.seen_urls:
                    continue
                
                self.seen_urls.add(art_url)
                
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('seendate', ''),
                    'url': art_url,
                    'source': article.get('domain', 'GDELT'),
                    'published_at': article.get('seendate', ''),
                    'content': '',
                })
            
            self.stats['gdelt'] += len(articles)
            logger.info(f"    Found: {len(articles)} articles")
            
        except Exception as e:
            logger.debug(f"    GDELT error: {e}")
        
        return articles
    
    def scrape_newsapi(self, topic: str, max_articles: int = 50) -> List[Dict]:
        """Scrape NewsAPI - Requires API key (free tier: 100 req/day)"""
        if not NEWS_API_KEY:
            return []
        
        logger.info(f"  NewsAPI: {topic[:40]}...")
        
        articles = []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': topic,
            'apiKey': NEWS_API_KEY,
            'pageSize': min(max_articles, 100),
            'sortBy': 'relevancy',
            'language': 'en',
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.warning(f"    NewsAPI error: {data.get('message', 'Unknown')}")
                return []
            
            for article in data.get('articles', []):
                art_url = article.get('url', '')
                
                if not art_url or art_url in self.seen_urls:
                    continue
                
                self.seen_urls.add(art_url)
                
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': art_url,
                    'source': article.get('source', {}).get('name', 'NewsAPI'),
                    'published_at': article.get('publishedAt', ''),
                    'content': article.get('content', ''),
                })
            
            self.stats['newsapi'] += len(articles)
            logger.info(f"    Found: {len(articles)} articles")
            
        except Exception as e:
            logger.debug(f"    NewsAPI error: {e}")
        
        return articles
    
    def scrape_tech_blogs(self) -> List[Dict]:
        """Scrape ML/AI focused tech blogs RSS feeds"""
        logger.info("  Scraping tech blog RSS feeds...")
        
        articles = []
        
        # ML/AI focused RSS feeds
        feeds = [
            ('https://distill.pub/rss.xml', 'Distill'),
            ('https://bair.berkeley.edu/blog/feed.xml', 'BAIR Blog'),
            ('https://ai.googleblog.com/feeds/posts/default', 'Google AI Blog'),
            ('https://openai.com/blog/rss/', 'OpenAI Blog'),
            ('https://www.deepmind.com/blog/rss.xml', 'DeepMind Blog'),
            ('https://lilianweng.github.io/index.xml', 'Lil\'s Log'),
            ('http://www.inference.vc/rss/', 'inFERENCe'),
            ('https://ruder.io/rss/index.rss', 'Sebastian Ruder'),
        ]
        
        uncertainty_keywords = [
            'uncertainty', 'bayesian', 'probabilistic', 'calibration',
            'confidence', 'ensemble', 'dropout', 'gaussian process',
            'prediction interval', 'out-of-distribution'
        ]
        
        for feed_url, source_name in feeds:
            try:
                response = self.session.get(feed_url, timeout=10)
                soup = BeautifulSoup(response.content, 'xml')
                
                items = soup.find_all('item') or soup.find_all('entry')
                
                for item in items:
                    # Get title and content
                    title = item.find('title')
                    title_text = title.text if title else ''
                    
                    desc = item.find('description') or item.find('summary') or item.find('content')
                    desc_text = desc.text if desc else ''
                    
                    # Check if uncertainty-related
                    combined = (title_text + ' ' + desc_text).lower()
                    if not any(kw in combined for kw in uncertainty_keywords):
                        continue
                    
                    # Get URL
                    link = item.find('link')
                    if link:
                        url = link.get('href') or link.text
                    else:
                        continue
                    
                    if url in self.seen_urls:
                        continue
                    
                    self.seen_urls.add(url)
                    
                    # Get date
                    pub_date = item.find('pubDate') or item.find('published') or item.find('updated')
                    
                    articles.append({
                        'title': title_text,
                        'description': desc_text[:500] if desc_text else '',
                        'url': url,
                        'source': source_name,
                        'published_at': pub_date.text if pub_date else '',
                        'content': '',
                    })
                
                time.sleep(0.3)
                
            except Exception as e:
                logger.debug(f"    Feed error ({source_name}): {e}")
        
        self.stats['tech_blogs'] += len(articles)
        logger.info(f"    Found: {len(articles)} relevant blog posts")
        return articles
    
    def save_article(self, article: Dict) -> bool:
        """Save article to file"""
        try:
            # Clean title for filename
            title = article.get('title', 'untitled')
            clean_title = re.sub(r'[^\w\s-]', '', title)
            clean_title = re.sub(r'[-\s]+', '-', clean_title)[:80]
            
            # Get date
            date_str = article.get('published_at', '')[:10]
            if not date_str or len(date_str) < 8:
                date_str = datetime.now().strftime('%Y-%m-%d')
            
            filename = f"{date_str}_{clean_title}.txt"
            filepath = self.output_dir / filename
            
            # Handle duplicates
            counter = 1
            while filepath.exists():
                filename = f"{date_str}_{clean_title}_{counter}.txt"
                filepath = self.output_dir / filename
                counter += 1
            
            # Write file
            lines = [
                f"Title: {article['title']}",
                f"Source: {article['source']}",
                f"URL: {article['url']}",
                f"Published: {article['published_at']}",
                "",
                "=" * 80,
                "",
                "Description:",
                article.get('description', '') or '(no description)',
                "",
            ]
            
            if article.get('content'):
                lines.extend([
                    "Content:",
                    article['content'],
                ])
            
            filepath.write_text('\n'.join(lines), encoding='utf-8')
            return True
            
        except Exception as e:
            logger.debug(f"Save error: {e}")
            return False
    
    def run(self) -> List[Dict]:
        """Run complete scraping pipeline"""
        print(f"\n{'#'*70}")
        print(f"# News Scraper v2.0")
        print(f"# Target: {TARGET_ARTICLES} articles")
        print(f"{'#'*70}")
        print(f"Topics: {len(TOPICS)}")
        print(f"Output: {self.output_dir}")
        print(f"{'#'*70}\n")
        
        # 1. Tech blogs first (highest quality)
        logger.info("\n[1/4] Tech Blogs")
        blog_articles = self.scrape_tech_blogs()
        for article in blog_articles:
            self.articles.append(article)
            self.save_article(article)
        
        # 2. Process each topic
        for idx, topic in enumerate(TOPICS, 1):
            if len(self.articles) >= TARGET_ARTICLES:
                break
            
            logger.info(f"\n[Topic {idx}/{len(TOPICS)}] {topic}")
            
            # Google News RSS
            rss_articles = self.scrape_google_news_rss(topic, max_articles=80)
            for article in rss_articles:
                if len(self.articles) >= TARGET_ARTICLES:
                    break
                article['topic'] = topic
                self.articles.append(article)
                self.save_article(article)
            
            # GDELT
            if len(self.articles) < TARGET_ARTICLES:
                gdelt_articles = self.scrape_gdelt(topic, max_articles=60)
                for article in gdelt_articles:
                    if len(self.articles) >= TARGET_ARTICLES:
                        break
                    article['topic'] = topic
                    self.articles.append(article)
                    self.save_article(article)
            
            # NewsAPI (if key available)
            if NEWS_API_KEY and len(self.articles) < TARGET_ARTICLES:
                newsapi_articles = self.scrape_newsapi(topic, max_articles=30)
                for article in newsapi_articles:
                    if len(self.articles) >= TARGET_ARTICLES:
                        break
                    article['topic'] = topic
                    self.articles.append(article)
                    self.save_article(article)
            
            logger.info(f"  Total: {len(self.articles)}/{TARGET_ARTICLES}")
            time.sleep(1)
        
        # Summary
        print(f"\n{'#'*70}")
        print(f"# SCRAPING COMPLETE")
        print(f"{'#'*70}")
        print(f"\nSources:")
        print(f"  Google News: {self.stats['google_news']}")
        print(f"  GDELT:       {self.stats['gdelt']}")
        print(f"  NewsAPI:     {self.stats['newsapi']}")
        print(f"  Tech Blogs:  {self.stats['tech_blogs']}")
        print(f"  {'─'*30}")
        print(f"  Total:       {len(self.articles)}")
        print(f"\nOutput: {self.output_dir}")
        print(f"Files: {len(list(self.output_dir.glob('*.txt')))}")
        print()
        
        return self.articles

# ============================================================================
# Main
# ============================================================================
def main():
    scraper = NewsScraper()
    scraper.run()

if __name__ == "__main__":
    main()