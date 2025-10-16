import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import shutil
import numpy as np
import json

# ------------------------
output_folder = r"cluster_output"

# 可视化文件夹，与 cluster_output 同级
vis_folder = os.path.join(os.path.dirname(output_folder), "cluster_visualizations")
os.makedirs(vis_folder, exist_ok=True)
embeddings_file = os.path.join(output_folder, "embeddings.pkl")


# ------------------------ 清理旧文件 ------------------------
if os.path.exists(vis_folder):
    shutil.rmtree(vis_folder)
os.makedirs(vis_folder, exist_ok=True)


# ------------------------
with open(embeddings_file, "rb") as f:
    embeddings = pickle.load(f)

# ------------------------
cluster_files = [f for f in os.listdir(output_folder) if f.endswith("_clusters.csv")]
print("Found cluster files:", cluster_files)

# ------------------------
# 降维到 50D，统一维度
pca50 = PCA(n_components=50, random_state=42)
reduced_embeddings = pca50.fit_transform(embeddings)

# ------------------------
for cluster_file in cluster_files:
    cluster_path = os.path.join(output_folder, cluster_file)
    df = pd.read_csv(cluster_path)
    labels = df['cluster'].to_numpy()
    if len(labels) != len(embeddings):
        print(f"Warning: {cluster_file} labels length mismatch, skipping.")
        continue

    base_name = os.path.splitext(cluster_file)[0]

    # ---- 加载 meta.json 提取簇代表文件名 ----
    meta_file = os.path.join(output_folder, f"{base_name}_meta.json")
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cluster_reps = {c: [m['filename'] for m in meta if m['cluster']==c][0] 
                        for c in set(labels) if c!=-1}
    else:
        cluster_reps = {}

    # ---- 绘图函数 ----
    def plot_2d(coords, title, save_name):
        plt.figure(figsize=(10,8))
        scatter = plt.scatter(coords[:,0], coords[:,1], c=labels, cmap='tab10', alpha=0.7, s=40)
        # 标注簇中心
        for c, fname in cluster_reps.items():
            idx = np.where(labels==c)[0]
            x_mean, y_mean = np.mean(coords[idx,0]), np.mean(coords[idx,1])
            plt.text(x_mean, y_mean, fname, fontsize=9, fontweight='bold')
        plt.title(f"{title} - {base_name}")
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(vis_folder, save_name), dpi=300)
        plt.close()

    # ---- PCA 2D ----
    pca2 = PCA(n_components=2, random_state=42)
    pca_2d = pca2.fit_transform(reduced_embeddings)
    plot_2d(pca_2d, "PCA 2D", f"{base_name}_pca2d.png")

    # ---- t-SNE ----
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_2d = tsne.fit_transform(reduced_embeddings)
    plot_2d(tsne_2d, "t-SNE", f"{base_name}_tsne.png")

    # ---- UMAP ----
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_2d = reducer.fit_transform(reduced_embeddings)
    plot_2d(umap_2d, "UMAP", f"{base_name}_umap.png")

    print(f"✅ Plots saved in '{vis_folder}' for {cluster_file}")
