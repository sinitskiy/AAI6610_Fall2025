#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reddit + StackExchange Crawler
Adapted to whole_pipeline project - Only changed paths
"""

import os
import re
import time
import glob
import praw
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

# ========== Path Configuration (Modified Section) ==========
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRAPER_OUTPUTS = PROJECT_ROOT / "scrapers" / "outputs"

# Output folder
OUTPUT_DIR = SCRAPER_OUTPUTS / "reddit_posts"
# ==========================================

# Clear old files before each run
if OUTPUT_DIR.exists():
    for f in OUTPUT_DIR.glob("*.txt"):
        f.unlink()
else:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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



# ----------------------------
# Reddit API Configuration (Unchanged)
# ----------------------------
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT")

# Search keywords and subreddit
SEARCH_KEYWORDS = [
    "uncertainty in machine learning",
    "model uncertainty",
    "uncertainty quantification",
    "antibody prediction",
    "antibody machine learning"
]

SUBREDDITS = [
    "MachineLearning",
    "ArtificialIntelligence",
    "biology",
    "bioinformatics",
    "datascience",
    "computervision"
]

# StackExchange sites
STACKEXCHANGE_SITES = [
    "datascience.stackexchange.com",
    "stats.stackexchange.com"
]

# ----------------------------
# Utility Functions
# ----------------------------
def sanitize_filename(text):
    return re.sub(r'[\\/*?:"<>|]', "_", text)[:120]

def save_post_to_txt(title, content, comments, url, source, date=None):
    if not title.strip():
        title = "(no title)"
    filename = sanitize_filename(title) + ".txt"
    filepath = OUTPUT_DIR / filename  # Modified: Use Path object
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Source: {source}\n")
        f.write(f"URL: {url}\n")
        if date:
            f.write(f"Date: {date}\n")
        f.write(f"Title: {title}\n\n")
        f.write(f"Content:\n{content}\n\n")
        f.write("Comments:\n")
        for c in comments:
            f.write(f"- {c}\n")
    print(f"Saved: {filename}")

# ----------------------------
# Scrape Reddit
# ----------------------------
def scrape_reddit():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    for sub in SUBREDDITS:
        subreddit = reddit.subreddit(sub)
        for kw in SEARCH_KEYWORDS:
            print(f"\nSearching Reddit: r/{sub} for '{kw}'")
            try:
                results = subreddit.search(kw, sort="new", limit=200)
                for post in results:
                    title = post.title
                    content = post.selftext
                    url = f"https://reddit.com{post.permalink}"
                    try:
                        date = datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        date = None
                    # Get comments
                    comments = []
                    try:
                        post.comments.replace_more(limit=0)
                        for c in post.comments.list():
                            if len(comments) >= 20:
                                break
                            comments.append(c.body.strip())
                    except Exception:
                        pass
                    save_post_to_txt(title, content, comments, url, f"Reddit/{sub}", date)
                    time.sleep(1.5)
            except Exception as e:
                print(f"Reddit error: {e}")
            time.sleep(3)

# ----------------------------
# Scrape StackExchange
# ----------------------------
def scrape_stackexchange():
    headers = {"User-Agent": "Mozilla/5.0"}
    for site in STACKEXCHANGE_SITES:
        for kw in SEARCH_KEYWORDS:
            print(f"\nSearching {site} for '{kw}'")
            for page in range(1, 6):  # First 5 pages
                search_url = f"https://{site}/search?page={page}&tab=newest&q={kw.replace(' ', '+')}"
                try:
                    resp = requests.get(search_url, headers=headers, timeout=15)
                    soup = BeautifulSoup(resp.text, "html.parser")
                    questions = soup.select("a.question-hyperlink")
                    if not questions:
                        break
                    for q in questions:
                        q_title = q.text.strip()
                        q_url = "https://" + site + q.get("href")
                        q_resp = requests.get(q_url, headers=headers, timeout=15)
                        q_soup = BeautifulSoup(q_resp.text, "html.parser")
                        q_content = q_soup.select_one(".js-post-body")
                        content = q_content.get_text("\n").strip() if q_content else ""
                        answers = [a.get_text("\n").strip() for a in q_soup.select(".answer .js-post-body")]
                        try:
                            time_elem = q_soup.select_one("time.relativetime")
                            if time_elem and time_elem.get("datetime"):
                                date = datetime.fromisoformat(time_elem.get("datetime").replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S")
                            else:
                                date = None
                        except:
                            date = None
                        save_post_to_txt(q_title, content, answers, q_url, f"StackExchange/{site}", date)
                        time.sleep(1)
                except Exception as e:
                    print(f"StackExchange error: {e}")
                    continue
                time.sleep(2)

# ----------------------------
# Main Program
# ----------------------------
if __name__ == "__main__":
    print("Starting Reddit + StackExchange scraping...\n")
    scrape_reddit()
    print("\nReddit scraping completed.\n")
    scrape_stackexchange()
    print(f"\nAll scraping completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total files scraped: {len(list(OUTPUT_DIR.glob('*.txt')))}\n")
