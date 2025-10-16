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

# ------------------------
# 配置
# -import os
import time
import json
import random
import hashlib
import requests
import shutil
from datetime import datetime
from playwright.sync_api import sync_playwright

# -----------------------------
# settings
# -----------------------------
GOOGLE_API_KEY = "-"   # Google Custom Search API Key
SEARCH_ENGINE_ID = "-"                       # Google Custom Search Engine ID

# 支持多个关键词
KEYWORDS = [
    "Applied to uncertainty prediction in ML models",
    "uncertainty estimation deep learning",
    "probabilistic modeling LLM",
    "Bayesian neural networks uncertainty",
    "epistemic aleatoric uncertainty"
]

NUM_TOTAL_PER_KEYWORD = 100      # 每个关键词抓取 100 条
LINKEDIN_COOKIES_FILE = "linkedin_cookies.json"
POST_FOLDER = "linkedin_posts"
MIN_DELAY = 3
MAX_DELAY = 8

# -----------------------------
# Google 搜索部分
# -----------------------------
def get_linkedin_urls(query, total=100):
    urls = []
    for start in range(1, total + 1, 10):  # 每次 10 条
        params = {
            "key": GOOGLE_API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": f"{query} site:linkedin.com/posts",
            "num": min(10, total - len(urls)),
            "start": start
        }
        resp = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        if resp.status_code != 200:
            print(f"❌ 请求 Google API 失败: {resp.status_code}")
            print(resp.text)
            break
        items = resp.json().get("items", [])
        urls.extend([item["link"] for item in items if "linkedin.com/posts" in item["link"]])
        time.sleep(random.randint(MIN_DELAY, MAX_DELAY))
        if len(urls) >= total:
            break
    return urls[:total]

# -----------------------------
# URL -> 唯一文件名
# -----------------------------
def url_to_filename(url):
    hash_id = hashlib.md5(url.encode("utf-8")).hexdigest()
    return f"{hash_id}.txt"

# -----------------------------
# Playwright 抓取内容
# -----------------------------
def scrape_post(url, folder=POST_FOLDER):
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, url_to_filename(url))
    if os.path.exists(filename):
        print(f"🟡 已抓取，跳过: {url}")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        if os.path.exists(LINKEDIN_COOKIES_FILE):
            with open(LINKEDIN_COOKIES_FILE, "r", encoding="utf-8") as f:
                cookies = json.load(f)
                context.add_cookies(cookies)
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(random.randint(2000, 4000))
        except Exception as e:
            print(f"❌ 访问失败: {url}", e)
            browser.close()
            return

        # 获取帖子内容
        try:
            text_content = page.inner_text("div.feed-shared-update-v2__description")
        except:
            text_content = page.inner_text("body")[:2000]

        # 日期
        try:
            date_text = page.inner_text("span.feed-shared-actor__sub-description > span > span")
        except:
            date_text = datetime.now().strftime("%Y-%m-%d")

        # 作者
        try:
            author_text = page.inner_text("span.feed-shared-actor__name")
        except:
            author_text = "Unknown"

        # 保存文本
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\nDate: {date_text}\nAuthor: {author_text}\n\n{text_content}")
        print(f"✅ Saved: {filename}")

        time.sleep(random.randint(MIN_DELAY, MAX_DELAY))
        browser.close()

# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    # 清空旧文件夹
    if os.path.exists(POST_FOLDER):
        shutil.rmtree(POST_FOLDER)
        print(f"🧹 已清空旧文件夹 {POST_FOLDER}")
    os.makedirs(POST_FOLDER, exist_ok=True)

    all_urls = set()
    for kw in KEYWORDS:
        print(f"\n==== 正在搜索关键词: {kw} ====")
        urls = get_linkedin_urls(kw, NUM_TOTAL_PER_KEYWORD)
        print(f"{kw} 抓到 {len(urls)} 条 URL")
        all_urls.update(urls)

    print(f"\n✅ 共合计去重后 {len(all_urls)} 条 URL")
    all_urls = list(all_urls)

    # 抓取内容
    for i, url in enumerate(all_urls, 1):
        print(f"\n[{i}/{len(all_urls)}] 抓取: {url}")
        try:
            scrape_post(url)
        except Exception as e:
            print("❌ 抓取失败:", url, e)
-----------------------
folder_path = r"linkedin_posts"
output_folder = r"cluster_output"
openai.api_key = "-"

batch_size = 10
top_keywords = 10
k_range = range(2, 11)  # KMeans 尝试 K=2~10

# ------------------------
# 清理旧文件
# ------------------------
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# ------------------------
# 文本预处理
# ------------------------
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
custom_stopwords = set(['2025', '10', 'author', 'url', 'unknown', 'date'])

def clean_text(text):
    text = re.sub(r"http\S+|<.*?>|跳到主要内容|领英热门内容|会员|领英学习|职位|游戏下载 APP|马上加入|登录", " ", text)
    text = unescape(text)
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = ' '.join([lemmatizer.lemmatize(w.lower()) 
                     for w in text.split() 
                     if w.lower() not in stop_words and w.lower() not in custom_stopwords])
    return text

# ------------------------
# 读取文本
# ------------------------
texts, file_names = [], []
for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            content = f.read()
            texts.append(clean_text(content))
            file_names.append(file)
print(f"Total texts loaded: {len(texts)}")

# ------------------------
# 生成 embeddings
# ------------------------
def get_embeddings(text_list, model="text-embedding-3-large"):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        response = openai.embeddings.create(model=model, input=batch)
        embeddings.extend([item.embedding for item in response.data])
        print(f"Batch {i//batch_size + 1}/{(len(text_list)+batch_size-1)//batch_size} done")
    return embeddings

embeddings = get_embeddings(texts)
with open(os.path.join(output_folder, "embeddings.pkl"), "wb") as f:
    pickle.dump(embeddings, f)

# ------------------------
# UMAP 降维到 50D
# ------------------------
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=50, metric='cosine', random_state=42)
umap_embeds = reducer.fit_transform(embeddings)

# ------------------------
# HDBSCAN 聚类
# ------------------------
hdb = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric='euclidean')
labels_hdb = hdb.fit_predict(umap_embeds)

# ------------------------
# KMeans 自动选最优 K
# ------------------------
best_sil = -1
best_k = 0
best_labels_km = None
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(umap_embeds)
    sil = silhouette_score(umap_embeds, labels)
    print(f"KMeans k={k}, silhouette={sil:.3f}")
    if sil > best_sil:
        best_sil = sil
        best_k = k
        best_labels_km = labels
print(f"✅ Best KMeans K={best_k}, silhouette={best_sil:.3f}")

# ------------------------
# 聚类质量指标函数
# ------------------------
def print_metrics(labels, name):
    mask = labels >= 0
    if len(set(labels[mask])) > 1:
        sil = silhouette_score(umap_embeds[mask], labels[mask])
        ch = calinski_harabasz_score(umap_embeds[mask], labels[mask])
        db = davies_bouldin_score(umap_embeds[mask], labels[mask])
        print(f"{name} metrics: silhouette={sil:.3f}, CH={ch:.3f}, DB={db:.3f}")
    else:
        print(f"{name}: Not enough clusters to calculate metrics.")

print_metrics(labels_hdb, "HDBSCAN")
print_metrics(best_labels_km, f"KMeans (K={best_k})")

# ------------------------
# UMAP 2D 可视化
# ------------------------
reducer2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
umap_2d = reducer2d.fit_transform(embeddings)

plt.figure(figsize=(7,5))
plt.scatter(umap_2d[:,0], umap_2d[:,1], c=[c if c>=0 else -1 for c in labels_hdb],
            cmap='tab20', s=10)
plt.colorbar(label='HDBSCAN Cluster (-1=noise)')
plt.title("HDBSCAN Clustering (UMAP 2D)")
plt.savefig(os.path.join(output_folder, "hdbscan_umap2d.png"))
plt.close()

plt.figure(figsize=(7,5))
plt.scatter(umap_2d[:,0], umap_2d[:,1], c=best_labels_km,
            cmap='tab20', s=10)
plt.colorbar(label=f'KMeans Cluster (K={best_k})')
plt.title("KMeans Clustering (UMAP 2D)")
plt.savefig(os.path.join(output_folder, "kmeans_umap2d.png"))
plt.close()

# ------------------------
# 保存代表文本和关键词
# ------------------------
def save_cluster_results(labels, model_name):
    df_clusters = pd.DataFrame({"filename": file_names, "cluster": labels})
    df_clusters.to_csv(os.path.join(output_folder, f"{model_name}_clusters.csv"), index=False, encoding="utf-8-sig")

    representatives = []
    meta_list = []
    for cluster_idx in set(labels):
        if cluster_idx == -1:
            continue
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster_idx]
        cluster_texts = [texts[i] for i in cluster_indices]

        rep_text = cluster_texts[0]
        rep_file = file_names[cluster_indices[0]]

        vectorizer = CountVectorizer(stop_words=list(stop_words.union(custom_stopwords)), max_features=1000)
        X = vectorizer.fit_transform(cluster_texts)
        keywords = np.array(vectorizer.get_feature_names_out())[X.sum(axis=0).A1.argsort()[-top_keywords:][::-1]]

        representatives.append(f"Cluster {cluster_idx} ({len(cluster_texts)} texts)\n"
                               f"Keywords: {', '.join(keywords)}\n"
                               f"Representative File: {rep_file}\n"
                               f"Abstract:\n{rep_text}\n\n")
        for idx in cluster_indices:
            meta_list.append({
                "filename": file_names[idx],
                "text": texts[idx],
                "cluster": int(cluster_idx),
                "keywords": ", ".join(keywords)
            })

    with open(os.path.join(output_folder, f"{model_name}_representatives.txt"), "w", encoding="utf-8") as f:
        f.writelines(representatives)
    with open(os.path.join(output_folder, f"{model_name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2, ensure_ascii=False)

save_cluster_results(labels_hdb, "HDBSCAN")
save_cluster_results(best_labels_km, f"KMeans_K{best_k}")

print(f"\n✅ Final clustering with HDBSCAN and KMeans complete! Results saved in {output_folder}")
