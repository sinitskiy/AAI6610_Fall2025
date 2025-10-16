import os
import re
import time
import glob
import praw
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ----------------------------
# 配置
# ----------------------------
OUTPUT_DIR = "Reddit_posts"

# 每次运行前清空旧文件
if os.path.exists(OUTPUT_DIR):
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*.txt")):
        os.remove(f)
else:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reddit API
REDDIT_CLIENT_ID = "-"
REDDIT_CLIENT_SECRET = "-"
REDDIT_USER_AGENT = "-"

# 搜索关键词和 subreddit
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

# StackExchange 站点
STACKEXCHANGE_SITES = [
    "datascience.stackexchange.com",
    "stats.stackexchange.com"
]

# ----------------------------
# 工具函数
# ----------------------------
def sanitize_filename(text):
    return re.sub(r'[\\/*?:"<>|]', "_", text)[:120]

def save_post_to_txt(title, content, comments, url, source, date=None):
    if not title.strip():
        title = "(no title)"
    filename = sanitize_filename(title) + ".txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Source: {source}\n")
        f.write(f"URL: {url}\n")
        if date:                      # ← 添加
            f.write(f"Date: {date}\n")
        f.write(f"Title: {title}\n\n")
        f.write(f"Content:\n{content}\n\n")
        f.write("Comments:\n")
        for c in comments:
            f.write(f"- {c}\n")
    print(f"✅ Saved: {filename}")

# ----------------------------
# 抓取 Reddit
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
            print(f"\n🔍 Searching Reddit: r/{sub} for '{kw}'")
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
                    # 获取评论
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
                print(f"❌ Reddit error: {e}")
            time.sleep(3)

# ----------------------------
# 抓取 StackExchange
# ----------------------------
def scrape_stackexchange():
    headers = {"User-Agent": "Mozilla/5.0"}
    for site in STACKEXCHANGE_SITES:
        for kw in SEARCH_KEYWORDS:
            print(f"\n🌐 Searching {site} for '{kw}'")
            for page in range(1, 6):  # 前5页
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
                    print(f"❌ StackExchange error: {e}")
                    continue
                time.sleep(2)

# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    print("🚀 开始抓取 Reddit + StackExchange 帖子...\n")
    scrape_reddit()
    print("\n✅ Reddit 抓取完成。\n")
    scrape_stackexchange()
    print("\n✅ 所有抓取完成！结果保存在 posts_txt 文件夹中。")
