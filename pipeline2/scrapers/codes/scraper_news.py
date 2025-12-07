#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized News Scraper - Target 1000 articles
Adapted to whole_pipeline project
"""

import os
import json
import time
import re
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import feedparser
import logging

# ========== Path Configuration ==========
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRAPER_OUTPUTS = PROJECT_ROOT / "scrapers" / "outputs"
DEFAULT_OUTPUT_DIR = SCRAPER_OUTPUTS / "news_articles"

# ========== Load Environment Variables ==========
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# ========== Configuration ==========
# NewsAPI configuration (optional, has free tier)
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "13c6ef9513ee45a88300ca29d5fa9855")

# Search topics
TOPICS = [
    "uncertainty quantification machine learning",
    "bayesian deep learning",
    "predictive uncertainty",
    "epistemic uncertainty",
    "aleatoric uncertainty",
    "confidence calibration",
    "probabilistic prediction",
    "monte carlo dropout",
    "ensemble uncertainty",
    "neural network uncertainty"
]

TARGET_ARTICLES = 1000

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NewsScraper:
    """News scraper - Multi-source scraping"""
    
    def __init__(self, output_dir=None, delay=0.5):
        """Initialize"""
        self.output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
        self.delay = delay
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.seen_urls = set()
        self.articles = []
    
    def scrape_google_news_rss(self, topic: str, max_per_topic: int = 150) -> list:
        """Scrape via Google News RSS - Completely free"""
        print(f"\nGoogle News RSS: {topic}")
        
        articles = []
        
        # More search variations
        search_variations = [
            topic,
            f"{topic} news",
            f"{topic} research",
            f"{topic} latest",
            f"{topic} analysis",
            f"{topic} developments",
            f"{topic} advances",
            f"{topic} applications"
        ]
        
        for search_term in search_variations:
            if len(articles) >= max_per_topic:
                break
            
            rss_url = f"https://news.google.com/rss/search?q={search_term}&hl=en&gl=US&ceid=US:en"
            
            try:
                response = self.session.get(rss_url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                print(f"   Found {len(items)} items for '{search_term}'")
                
                for item in items:
                    url = item.find('link').text if item.find('link') else ''
                    
                    if url in self.seen_urls:
                        continue
                    self.seen_urls.add(url)
                    
                    articles.append({
                        'title': item.find('title').text if item.find('title') else '',
                        'description': item.find('description').text if item.find('description') else '',
                        'url': url,
                        'source': 'Google News',
                        'published_at': item.find('pubDate').text if item.find('pubDate') else '',
                    })
                
                time.sleep(0.5)  # Reduced delay for faster speed
                
            except Exception as e:
                print(f"   Error: {e}")
        
        print(f"   Collected {len(articles)} unique articles")
        return articles[:max_per_topic]
    
    def scrape_newsapi(self, topic: str, max_articles: int = 50) -> list:
        """Scrape using NewsAPI - Free tier has 100 requests per day"""
        print(f"\nNewsAPI: {topic}")
        
        if not NEWS_API_KEY:
            print("   Warning: NewsAPI key not set, skipping")
            return []
        
        articles = []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': topic,
            'apiKey': NEWS_API_KEY,
            'pageSize': min(max_articles, 100),
            'sortBy': 'relevancy',
            'language': 'en'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            for article in data.get('articles', []):
                url = article.get('url', '')
                
                if url in self.seen_urls:
                    continue
                self.seen_urls.add(url)
                
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': url,
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'content': article.get('content', '')
                })
            
            print(f"   Collected {len(articles)} articles")
            
        except Exception as e:
            print(f"   Error: {e}")
        
        return articles
    
    def scrape_gdelt(self, topic: str, max_articles: int = 100) -> list:
        """Use GDELT - Completely free, no limits"""
        print(f"\nGDELT: {topic}")
        
        articles = []
        
        gdelt_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            'query': topic,
            'mode': 'artlist',
            'maxrecords': max_articles,
            'format': 'json'
        }
        
        try:
            response = self.session.get(gdelt_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            for article in data.get('articles', []):
                url = article.get('url', '')
                
                if url in self.seen_urls:
                    continue
                self.seen_urls.add(url)
                
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('seendate', ''),
                    'url': url,
                    'source': article.get('domain', ''),
                    'published_at': article.get('seendate', ''),
                    'content': ''
                })
            
            print(f"   Collected {len(articles)} articles")
            
        except Exception as e:
            print(f"   Error: {e}")
        
        return articles
    
    def save_article(self, article: dict):
        """Save a single article"""
        try:
            # Generate filename
            title = re.sub(r'[^\w\s-]', '', article['title'])
            title = re.sub(r'[-\s]+', '-', title)[:100]
            
            date_str = article.get('published_at', datetime.now().isoformat())[:10]
            filename = f"{date_str}_{title}.txt"
            filepath = self.output_dir / filename
            
            # Avoid duplicates
            counter = 1
            while filepath.exists():
                filepath = self.output_dir / f"{date_str}_{title}_{counter}.txt"
                counter += 1
            
            # Save
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {article['title']}\n")
                f.write(f"Source: {article['source']}\n")
                f.write(f"URL: {article['url']}\n")
                f.write(f"Published: {article['published_at']}\n")
                f.write("="*80 + "\n\n")
                f.write(f"Description:\n{article['description']}\n\n")
                
                if article.get('content'):
                    f.write(f"Content:\n{article['content']}\n")
            
        except Exception as e:
            print(f"   Error saving: {e}")
    
    def run(self):
        """Run complete scraping - Optimized for 1000 articles"""
        print(f"\n{'='*70}")
        print("News Scraper (Target: 1000 articles)")
        print(f"{'='*70}")
        print(f"Topics: {len(TOPICS)}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")
        
        for topic in TOPICS:
            if len(self.articles) >= TARGET_ARTICLES:
                break
            
            print(f"\n{'─'*70}")
            print(f"Topic: {topic}")
            print(f"{'─'*70}")
            
            # Google News RSS (about 150 articles per topic)
            articles_rss = self.scrape_google_news_rss(topic, max_per_topic=150)
            
            for article in articles_rss:
                if len(self.articles) >= TARGET_ARTICLES:
                    break
                
                article['topic'] = topic
                self.articles.append(article)
                self.save_article(article)
            
            # GDELT (about 100 articles per topic)
            if len(self.articles) < TARGET_ARTICLES:
                articles_gdelt = self.scrape_gdelt(topic, max_articles=100)
                
                for article in articles_gdelt:
                    if len(self.articles) >= TARGET_ARTICLES:
                        break
                    
                    article['topic'] = topic
                    self.articles.append(article)
                    self.save_article(article)
            
            # NewsAPI (about 50 articles per topic, optional)
            if NEWS_API_KEY and len(self.articles) < TARGET_ARTICLES:
                articles_newsapi = self.scrape_newsapi(topic, max_articles=50)
                
                for article in articles_newsapi:
                    if len(self.articles) >= TARGET_ARTICLES:
                        break
                    
                    article['topic'] = topic
                    self.articles.append(article)
                    self.save_article(article)
            
            print(f"   Total collected so far: {len(self.articles)}/{TARGET_ARTICLES}")
        
        print(f"\n{'='*70}")
        print(f"News scraping complete!")
        print(f"{'='*70}")
        print(f"Total articles: {len(self.articles)}")
        print(f"Unique URLs: {len(self.seen_urls)}")
        print(f"Output: {self.output_dir}")
        print(f"Files saved: {len(list(self.output_dir.glob('*.txt')))}")
        print(f"{'='*70}\n")
        
        return self.articles


def main():
    """Main program"""
    print(f"\n{'='*70}")
    print("News Scraper (Target: 1000 articles)")
    print(f"{'='*70}\n")
    
    scraper = NewsScraper()
    articles = scraper.run()
    
    print(f"\n{'='*70}")
    print(f"Scraping complete!")
    print(f"{'='*70}")
    print(f"Articles: {len(articles)}")
    print(f"Output: {scraper.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
