#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LinkedIn Crawler - Adapted path to whole_pipeline
"""

import os
import re
import pickle
import json
import shutil
import pandas as pd
import openai
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan
import umap
import numpy as np
import matplotlib.pyplot as plt
from html import unescape
import time
import random
import hashlib
import requests
from datetime import datetime
from playwright.sync_api import sync_playwright
from pathlib import Path

# ========== Path Configuration (Modified Section) ==========
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRAPER_OUTPUTS = PROJECT_ROOT / "scrapers" / "outputs"

# Cookie file path
LINKEDIN_COOKIES_FILE = Path(__file__).parent / "linkedin_cookies.json"

# Output folder
POST_FOLDER = SCRAPER_OUTPUTS / "linkedin_posts"
# ==========================================

# ========== Load Environment Variables ==========
env_path = PROJECT_ROOT / ".env"
print(f"Loading .env from: {env_path}")

if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    print("Environment variables loaded")
else:
    print(f"Warning: .env file not found at {env_path}")

# ========== API Configuration ==========
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
SEARCH_ENGINE_ID = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "")


# Support multiple keywords
KEYWORDS = [
    "Applied to uncertainty prediction in ML models",
    "uncertainty estimation deep learning",
    "probabilistic modeling LLM",
    "Bayesian neural networks uncertainty",
    "epistemic aleatoric uncertainty"
]

NUM_TOTAL_PER_KEYWORD = 100  # Scrape 100 items per keyword
MIN_DELAY = 3
MAX_DELAY = 8

# -----------------------------
# Google Search Section
# -----------------------------
def get_linkedin_urls(query, total=100):
    urls = []
    for start in range(1, total + 1, 10):  # 10 items per batch
        params = {
            "key": GOOGLE_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": f"{query} site:linkedin.com/posts",
            "num": min(10, total - len(urls)),
            "start": start
        }
        resp = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        if resp.status_code != 200:
            print(f"Google API request failed: {resp.status_code}")
            print(resp.text)
            break
        items = resp.json().get("items", [])
        urls.extend([item["link"] for item in items if "linkedin.com/posts" in item["link"]])
        time.sleep(random.randint(MIN_DELAY, MAX_DELAY))
        if len(urls) >= total:
            break
    return urls[:total]

# -----------------------------
# URL -> Unique filename
# -----------------------------
def url_to_filename(url):
    hash_id = hashlib.md5(url.encode("utf-8")).hexdigest()
    return f"{hash_id}.txt"

# -----------------------------
# Playwright scraping content
# -----------------------------
def scrape_post(url, folder=POST_FOLDER):
    # Modified: Use Path object
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / url_to_filename(url)
    
    if filename.exists():
        print(f"Already scraped, skipping: {url}")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        
        # Modified: Check Path object
        if LINKEDIN_COOKIES_FILE.exists():
            with open(LINKEDIN_COOKIES_FILE, "r", encoding="utf-8") as f:
                cookies = json.load(f)
                context.add_cookies(cookies)
        
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(random.randint(2000, 4000))
        except Exception as e:
            print(f"Access failed: {url}", e)
            browser.close()
            return

        # Get post content
        try:
            text_content = page.inner_text("div.feed-shared-update-v2__description")
        except:
            text_content = page.inner_text("body")[:2000]

        # Date
        try:
            date_text = page.inner_text("span.feed-shared-actor__sub-description > span > span")
        except:
            date_text = datetime.now().strftime("%Y-%m-%d")

        # Author
        try:
            author_text = page.inner_text("span.feed-shared-actor__name")
        except:
            author_text = "Unknown"

        # Save text
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\nDate: {date_text}\nAuthor: {author_text}\n\n{text_content}")
        print(f"Saved: {filename.name}")

        time.sleep(random.randint(MIN_DELAY, MAX_DELAY))
        browser.close()

# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    # Clear old folder
    if POST_FOLDER.exists():
        shutil.rmtree(POST_FOLDER)
        print(f"Cleaned old folder {POST_FOLDER}")
    POST_FOLDER.mkdir(parents=True, exist_ok=True)

    all_urls = set()
    for kw in KEYWORDS:
        print(f"\n==== Searching keyword: {kw} ====")
        urls = get_linkedin_urls(kw, NUM_TOTAL_PER_KEYWORD)
        print(f"{kw} found {len(urls)} URLs")
        all_urls.update(urls)

    print(f"\nTotal unique URLs after deduplication: {len(all_urls)}")
    all_urls = list(all_urls)

    # Scrape content
    for i, url in enumerate(all_urls, 1):
        print(f"\n[{i}/{len(all_urls)}] Scraping: {url}")
        try:
            scrape_post(url)
        except Exception as e:
            print("Scraping failed:", url, e)
    
    print(f"\n{'='*70}")
    print(f"LinkedIn scraping completed!")
    print(f"Output directory: {POST_FOLDER}")
    print(f"Total files scraped: {len(list(POST_FOLDER.glob('*.txt')))}")
    print(f"{'='*70}\n")
