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
import hdbscan
import umap
import numpy as np
import matplotlib.pyplot as plt
from html import unescape
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ------------------------ 配置 ------------------------
folder_path = r"filtered_posts"
output_folder = r"cluster_output"

# 保留你的 API Key
openai.api_key = "-"

batch_size = 10
top_keywords = 10
max_words_per_text = 500  # 拆分长文本，避免超过 8192 tokens

# ------------------------ 清理旧文件 ------------------------
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# ------------------------ 文本预处理 ------------------------
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
custom_stopwords = set([
    '2025', '10', 'author', 'url', 'unknown', 'date',
    'ai', 'data', 'learning', 'uncertainty', 'model', 'models'
])

def clean_text(text):
    text = re.sub(r"http\S+|<.*?>|跳到主要内容|领英热门内容|会员|领英学习|职位|游戏下载 APP|马上加入|登录", " ", text)
    text = unescape(text)
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = ' '.join([lemmatizer.lemmatize(w.lower()) for w in text.split()
                     if w.lower() not in stop_words and w.lower() not in custom_stopwords])
    return text

# ------------------------ 读取文本 ------------------------
texts, file_names = [], []
for file in os.listdir(folder_path):
    if file.endswith(".txt"):
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            content = f.read()
            texts.append(clean_text(content))
            file_names.append(file)

print(f"Total texts loaded: {len(texts)}")

# ------------------------ 拆分长文本 ------------------------
def split_long_text(text, max_words=max_words_per_text):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

texts_split, file_names_split = [], []
for t, fname in zip(texts, file_names):
    parts = split_long_text(t)
    texts_split.extend(parts)
    file_names_split.extend([fname]*len(parts))

# ------------------------ 生成 embeddings ------------------------
def get_embeddings(text_list, model="text-embedding-3-large"):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        response = openai.embeddings.create(model=model, input=batch)
        embeddings.extend([item.embedding for item in response.data])
        print(f"Batch {i//batch_size + 1}/{(len(text_list)+batch_size-1)//batch_size} done")
    return embeddings

embeddings = get_embeddings(texts_split)

with open(os.path.join(output_folder, "embeddings.pkl"), "wb") as f:
    pickle.dump(embeddings, f)

# ------------------------ UMAP 降维到 50D ------------------------
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=50, metric='cosine', random_state=42)
umap_embeds = reducer.fit_transform(embeddings)

# ------------------------ HDBSCAN 聚类 ------------------------
hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3, metric='euclidean')
labels_hdb = hdb.fit_predict(umap_embeds)

# ------------------------ KMeans 聚类 + 自动选择 K ------------------------
def best_kmeans(X, min_k=2, max_k=10):
    best_sil = -1
    best_labels = None
    best_k = min_k
    for k in range(min_k, max_k+1):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        print(f"KMeans k={k}, silhouette={sil:.3f}")
        if sil > best_sil:
            best_sil = sil
            best_labels = labels
            best_k = k
    print(f"✅ Best KMeans K={best_k}, silhouette={best_sil:.3f}")
    return best_labels, best_k, best_sil

kmeans_labels, best_k, best_sil = best_kmeans(umap_embeds)

# ------------------------ 聚类指标计算 ------------------------
def calc_metrics(X, labels, model_name):
    mask = labels != -1
    if mask.sum() <= 1:
        sil, ch, db = -1, -1, -1
    else:
        sil = silhouette_score(X[mask], labels[mask])
        ch = calinski_harabasz_score(X[mask], labels[mask])
        db = davies_bouldin_score(X[mask], labels[mask])
    print(f"{model_name} metrics: silhouette={sil:.3f}, CH={ch:.3f}, DB={db:.3f}")
    return sil, ch, db

sil_hdb, ch_hdb, db_hdb = calc_metrics(umap_embeds, labels_hdb, "HDBSCAN")
sil_km, ch_km, db_km = calc_metrics(umap_embeds, kmeans_labels, f"KMeans_K{best_k}")

# ------------------------ 保存结果 ------------------------
def save_cluster_results(labels, model_name):
    df_clusters = pd.DataFrame({"filename": file_names_split, "cluster": labels})
    df_clusters.to_csv(os.path.join(output_folder, f"{model_name}_clusters.csv"), index=False, encoding="utf-8-sig")

    representatives = []
    meta_list = []
    for cluster_idx in set(labels):
        if cluster_idx == -1:
            continue
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster_idx]
        cluster_texts = [texts_split[i] for i in cluster_indices]
        rep_text = cluster_texts[0]
        rep_file = file_names_split[cluster_indices[0]]

        vectorizer = CountVectorizer(stop_words=list(stop_words.union(custom_stopwords)), max_features=1000)
        X = vectorizer.fit_transform(cluster_texts)
        keywords = np.array(vectorizer.get_feature_names_out())[X.sum(axis=0).A1.argsort()[-top_keywords:][::-1]]

        representatives.append(f"Cluster {cluster_idx} ({len(cluster_texts)} texts)\n"
                               f"Keywords: {', '.join(keywords)}\n"
                               f"Representative File: {rep_file}\n"
                               f"Abstract:\n{rep_text}\n\n")
        for idx in cluster_indices:
            meta_list.append({
                "filename": file_names_split[idx],
                "text": texts_split[idx],
                "cluster": int(cluster_idx),
                "keywords": ", ".join(keywords)
            })

    with open(os.path.join(output_folder, f"{model_name}_representatives.txt"), "w", encoding="utf-8") as f:
        f.writelines(representatives)
    with open(os.path.join(output_folder, f"{model_name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2, ensure_ascii=False)

save_cluster_results(labels_hdb, "HDBSCAN")
save_cluster_results(kmeans_labels, f"KMeans_K{best_k}")

# ------------------------ 可视化 HDBSCAN ------------------------
reducer2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
umap_2d = reducer2d.fit_transform(embeddings)
plt.figure(figsize=(7,5))
plt.scatter(umap_2d[:,0], umap_2d[:,1], c=labels_hdb, cmap='tab20', s=10)
plt.colorbar(label='Cluster')
plt.title("HDBSCAN Clustering (UMAP 2D)")
plt.savefig(os.path.join(output_folder, "hdbscan_umap2d.png"))
plt.close()

plt.figure(figsize=(7,5))
plt.scatter(umap_2d[:,0], umap_2d[:,1], c=kmeans_labels, cmap='tab20', s=10)
plt.colorbar(label='Cluster')
plt.title(f"KMeans K={best_k} Clustering (UMAP 2D)")
plt.savefig(os.path.join(output_folder, f"kmeans_k{best_k}_umap2d.png"))
plt.close()

print(f"\n✅ Clustering complete! Results saved in {output_folder}")
