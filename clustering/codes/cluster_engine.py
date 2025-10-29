#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clustering Engine - Topic clustering for filtered texts
"""

import sys
import io

# Windows GBK encoding fix
#if sys.platform == 'win32':
    #sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    #sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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

# ============================================================================
# Configuration - Using relative paths
# ============================================================================
# Get script directory (clustering/codes/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root directory (AAI6610_WholePipeline/)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Input/output paths (clustering/outputs/)
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "..", "outputs")
INPUT_FOLDER = os.path.join(OUTPUT_ROOT, "filtered_posts_all_sources")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "cluster_output")



# Load .env file
from pathlib import Path
env_path = Path(__file__).parent.parent.parent / ".env"

if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Read API Key
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

# Validate
if not openai.api_key:
    print("OpenAI API key not found in .env!")
    exit(1)

print(f"OpenAI API Key loaded (first 20 chars): {openai.api_key[:20]}...")


# Clustering parameters
BATCH_SIZE = 10
TOP_KEYWORDS = 10
MAX_WORDS_PER_TEXT = 500

# ============================================================================
# Validate Paths
# ============================================================================
def validate_paths():
    """Validate input/output paths"""
    print(f"\nValidating paths...")
    print(f"   Script location: {SCRIPT_DIR}")
    print(f"   Project root:    {PROJECT_ROOT}")
    print(f"   Input folder:    {os.path.relpath(INPUT_FOLDER, PROJECT_ROOT)}")
    print(f"   Output folder:   {os.path.relpath(OUTPUT_FOLDER, PROJECT_ROOT)}\n")
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder not found!")
        print(f"   Expected: {INPUT_FOLDER}")
        return False
    
    file_count = len([f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')])
    print(f"Found {file_count} text files in input folder\n")
    
    return True

# ============================================================================
# Clean Old Output
# ============================================================================
def clean_output_folder():
    """Clean old output folder"""
    if os.path.exists(OUTPUT_FOLDER):
        print(f"Cleaning old output folder...")
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"   Created clean folder: {os.path.basename(OUTPUT_FOLDER)}/\n")

# ============================================================================
# Text Preprocessing
# ============================================================================
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
custom_stopwords = set([
    '2025', '10', 'author', 'url', 'unknown', 'date',
    'ai', 'data', 'learning', 'uncertainty', 'model', 'models',
    'linkedin', 'reddit', 'post', 'article', 'paper'
])

def clean_text(text):
    """Clean text"""
    # Remove LinkedIn/Reddit UI elements
    text = re.sub(r"http\S+|<.*?>|Jump to main content|LinkedIn hot content|Member|LinkedIn learning|Job|Game download APP|Join now|Log in", " ", text)
    text = unescape(text)
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Lemmatization and stopword filtering
    text = ' '.join([
        lemmatizer.lemmatize(w.lower()) 
        for w in text.split()
        if w.lower() not in stop_words and w.lower() not in custom_stopwords
    ])
    return text

# ============================================================================
# Extract Title and Abstract
# ============================================================================
def extract_title_and_preview(original_text, max_preview_chars=500):
    """Extract title and abstract preview from original text"""
    if not original_text or not original_text.strip():
        return "No Title", "No content available."
    
    lines = original_text.strip().split('\n')
    
    # Filter patterns
    ui_patterns = [
        'Jump to main content', 'LinkedIn', 'Hot content', 'Member', 'LinkedIn learning', 'Job', 
        'Game', 'Download APP', 'Join now', 'Log in', 'LinkedIn',
        'updates', 'month', 'hour', 'day', 'week', 'minute',
        'cookie', 'Cookie', 'career', 'productivity', 'finance',
        'Source:', 'URL:', 'Date:', 'Author:', 'Title:', 'Abstract:',
        'Published:', 'DOI/ID:', 'Categories:'
    ]
    
    job_patterns = ['CTO at', 'CEO at', 'Engineer at', 'Manager at']
    metadata_patterns = ['URL:', 'Date:', 'Author:', 'http://', 'https://']
    
    # Collect content lines
    content_lines = []
    for line in lines:
        line = line.strip()
        
        if not line or len(line) < 15:
            continue
        if any(line.startswith(p) for p in metadata_patterns):
            continue
        if any(p in line for p in ui_patterns):
            continue
        if any(p in line for p in job_patterns):
            continue
        
        content_lines.append(line)
    
    if not content_lines:
        return "No Title", "No content available."
    
    # Extract title
    title = content_lines[0]
    if len(title) > 150:
        title = title[:147] + "..."
    
    # Extract preview
    preview = ' '.join(content_lines[:5])
    if len(preview) > max_preview_chars:
        preview = preview[:max_preview_chars-3] + "..."
    
    return title, preview

# ============================================================================
# Read Texts
# ============================================================================
def read_texts():
    """Read all filtered texts"""
    print(f"Reading texts from {os.path.relpath(INPUT_FOLDER, PROJECT_ROOT)}/")
    
    texts_clean = []
    texts_original = []
    file_names = []
    
    txt_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')]
    
    for file in txt_files:
        try:
            file_path = os.path.join(INPUT_FOLDER, file)
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                content = f.read()
                
                if len(content.strip()) < 50:
                    continue
                
                texts_original.append(content)
                texts_clean.append(clean_text(content))
                file_names.append(file)
        except Exception as e:
            print(f"   Error reading {file}: {e}")
    
    print(f"   Loaded {len(texts_clean)} texts\n")
    
    # Create filename to original text mapping
    file_to_original = {fname: orig for fname, orig in zip(file_names, texts_original)}
    
    return texts_clean, texts_original, file_names, file_to_original

# ============================================================================
# Text Chunking
# ============================================================================
def split_long_text(text, max_words=MAX_WORDS_PER_TEXT):
    """Split overly long text"""
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def chunk_texts(texts_clean, file_names):
    """Chunk all texts"""
    print(f"Chunking long texts (max {MAX_WORDS_PER_TEXT} words per chunk)...")
    
    texts_split = []
    file_names_split = []
    
    for clean_txt, fname in zip(texts_clean, file_names):
        clean_parts = split_long_text(clean_txt)
        texts_split.extend(clean_parts)
        file_names_split.extend([fname] * len(clean_parts))
    
    print(f"   Created {len(texts_split)} chunks from {len(file_names)} files\n")
    
    return texts_split, file_names_split

# ============================================================================
# Generate Embeddings
# ============================================================================
def get_embeddings(text_list, model="text-embedding-3-large"):
    """Generate embeddings using OpenAI API"""
    print(f"Generating embeddings using {model}...")
    print(f"   Processing {len(text_list)} text chunks in batches of {BATCH_SIZE}...")
    
    embeddings = []
    total_batches = (len(text_list) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(text_list), BATCH_SIZE):
        batch = text_list[i:i+BATCH_SIZE]
        try:
            response = openai.embeddings.create(model=model, input=batch)
            embeddings.extend([item.embedding for item in response.data])
            print(f"      Batch {i//BATCH_SIZE + 1}/{total_batches} done")
        except Exception as e:
            print(f"      Error in batch {i//BATCH_SIZE + 1}: {e}")
            # Fill failed batch with zero vectors
            embeddings.extend([[0.0] * 3072] * len(batch))
    
    print(f"   Generated {len(embeddings)} embeddings\n")
    return embeddings

# ============================================================================
# Dimensionality Reduction and Clustering
# ============================================================================
def perform_clustering(embeddings):
    """Perform UMAP dimensionality reduction and clustering"""
    print(f"Performing dimensionality reduction and clustering...")
    
    # UMAP dimension reduction to 50D
    print(f"   Reducing dimensions with UMAP (3072 -> 50)...")
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        n_components=50, 
        metric='cosine', 
        random_state=42
    )
    umap_embeds = reducer.fit_transform(embeddings)
    print(f"   UMAP reduction complete")
    
    # HDBSCAN clustering
    print(f"   Clustering with HDBSCAN...")
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=10, 
        min_samples=3, 
        metric='euclidean'
    )
    labels_hdb = hdb.fit_predict(umap_embeds)
    n_clusters_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
    print(f"   HDBSCAN found {n_clusters_hdb} clusters")
    
    # KMeans clustering (automatically select optimal K)
    print(f"   Finding optimal K for KMeans...")
    kmeans_labels, best_k = find_best_kmeans(umap_embeds)
    print(f"   KMeans with K={best_k}\n")
    
    return umap_embeds, labels_hdb, kmeans_labels, best_k

def find_best_kmeans(X, min_k=5, max_k=20):
    """Automatically select optimal K value through silhouette coefficient"""
    best_sil = -1
    best_labels = None
    best_k = min_k
    
    for k in range(min_k, max_k+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        
        if sil > best_sil:
            best_sil = sil
            best_labels = labels
            best_k = k
    
    print(f"      Best K={best_k} with silhouette={best_sil:.3f}")
    return best_labels, best_k

# ============================================================================
# Calculate Clustering Metrics
# ============================================================================
def calculate_metrics(X, labels, model_name):
    """Calculate clustering quality metrics"""
    mask = labels != -1
    
    if mask.sum() <= 1:
        return -1, -1, -1
    
    sil = silhouette_score(X[mask], labels[mask])
    ch = calinski_harabasz_score(X[mask], labels[mask])
    db = davies_bouldin_score(X[mask], labels[mask])
    
    print(f"   {model_name}:")
    print(f"      Silhouette: {sil:.3f}")
    print(f"      Calinski-Harabasz: {ch:.1f}")
    print(f"      Davies-Bouldin: {db:.3f}")
    
    return sil, ch, db

# ============================================================================
# Cluster Summarization (New)
# ============================================================================
def summarize_cluster(cluster_texts_clean, keywords, model="gpt-4o-mini"):
    """Generate a human-readable summary of each cluster using OpenAI GPT."""
    try:
        joined_texts = "\n\n".join(cluster_texts_clean[:3])  # take up to 3 samples
        prompt = f"""
        You are an expert science and tech communicator.
        The following texts belong to one cluster discovered by topic clustering.
        Summarize the *main topic, theme, or trend* of this cluster in 2–3 human-readable sentences.
        Focus on what these texts are about, why they matter, and what connects them.
        
        Keywords: {', '.join(keywords)}
        
        Example texts:
        {joined_texts}
        """
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"Summarization error: {e}")
        return "Summary not available."

# ============================================================================
# Save Clustering Results
# ============================================================================
def save_cluster_results(labels, model_name, texts_split, file_names_split, 
                        file_to_original):
    """Save clustering results and representative documents"""
    print(f"\nSaving {model_name} results...")
    
    # Save cluster assignments
    df_clusters = pd.DataFrame({
        "filename": file_names_split, 
        "cluster": labels
    })
    csv_path = os.path.join(OUTPUT_FOLDER, f"{model_name}_clusters.csv")
    df_clusters.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"   Saved cluster assignments")
    
    # Generate summary for each cluster
    representatives = []
    meta_list = []
    
    for cluster_idx in sorted(set(labels)):
        if cluster_idx == -1:
            continue
        
        # Get all documents in this cluster
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster_idx]
        cluster_texts_clean = [texts_split[i] for i in cluster_indices]
        
        # Select representative document (first one)
        rep_idx = cluster_indices[0]
        rep_file = file_names_split[rep_idx]
        rep_text_original = file_to_original[rep_file]
        
        # Extract title and abstract
        title, preview = extract_title_and_preview(rep_text_original)
        
        # Extract keywords
        try:
            vectorizer = CountVectorizer(
                stop_words=list(stop_words.union(custom_stopwords)), 
                max_features=1000
            )
            X = vectorizer.fit_transform(cluster_texts_clean)
            keywords = np.array(vectorizer.get_feature_names_out())[
                X.sum(axis=0).A1.argsort()[-TOP_KEYWORDS:][::-1]
            ]
        except:
            keywords = []
        # Generate human-readable summary (NEW)
        summary = summarize_cluster(cluster_texts_clean, keywords)
        # Save representative document information
        representatives.append(
            f"{'='*70}\n"
            f"Cluster {cluster_idx} | Size: {len(cluster_texts_clean)} documents\n"
            f"{'='*70}\n"
            f"Representative File: {rep_file}\n\n"
            f"Title:\n{title}\n\n"
            f"Keywords:\n{', '.join(keywords)}\n\n"
            f"Cluster Summary:\n{summary}\n\n"
            f"Abstract Preview:\n{preview}\n\n"
        )
        
        # Save metadata
        for idx in cluster_indices:
            fname = file_names_split[idx]
            meta_list.append({
                "filename": fname,
                "text": texts_split[idx],
                "original_text": file_to_original[fname],
                "cluster": int(cluster_idx),
                "keywords": ", ".join(keywords)
            })
    
    # Save representative documents
    txt_path = os.path.join(OUTPUT_FOLDER, f"{model_name}_representatives.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(representatives)
    print(f"   Saved cluster representatives")
    
    # Save metadata JSON
    json_path = os.path.join(OUTPUT_FOLDER, f"{model_name}_meta.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2, ensure_ascii=False)
    print(f"   Saved metadata JSON")

# ============================================================================
# Visualization
# ============================================================================
def create_visualizations(embeddings, labels_hdb, kmeans_labels, best_k):
    """Generate clustering visualizations"""
    print(f"\nCreating visualizations...")
    
    # UMAP dimension reduction to 2D for visualization
    reducer2d = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        n_components=2, 
        metric='cosine', 
        random_state=42
    )
    umap_2d = reducer2d.fit_transform(embeddings)
    
    # HDBSCAN visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        umap_2d[:, 0], 
        umap_2d[:, 1], 
        c=labels_hdb, 
        cmap='tab20', 
        s=20, 
        alpha=0.6
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title("HDBSCAN Clustering (UMAP 2D)", fontsize=14, fontweight='bold')
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    hdb_path = os.path.join(OUTPUT_FOLDER, "hdbscan_umap2d.png")
    plt.savefig(hdb_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved HDBSCAN visualization")
    
    # KMeans visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        umap_2d[:, 0], 
        umap_2d[:, 1], 
        c=kmeans_labels, 
        cmap='tab20', 
        s=20, 
        alpha=0.6
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title(f"KMeans Clustering (K={best_k}, UMAP 2D)", fontsize=14, fontweight='bold')
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    km_path = os.path.join(OUTPUT_FOLDER, f"kmeans_k{best_k}_umap2d.png")
    plt.savefig(km_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved KMeans visualization")

# ============================================================================
# Main Function
# ============================================================================
def main():
    print(f"\n{'#'*70}")
    print(f"# PIPELINE 3: CLUSTERING ANALYSIS")
    print(f"# Uncertainty Estimation Topic Discovery")
    print(f"{'#'*70}\n")
    
    # Step 0: Validate paths
    if not validate_paths():
        return
    
    # Step 1: Clean output folder
    clean_output_folder()
    
    # Step 2: Read texts
    texts_clean, texts_original, file_names, file_to_original = read_texts()
    
    if len(texts_clean) == 0:
        print("No texts found!")
        return
    
    # Step 3: Text chunking
    texts_split, file_names_split = chunk_texts(texts_clean, file_names)
    
    # Step 4: Generate embeddings
    embeddings = get_embeddings(texts_split)
    
    # Save embeddings
    emb_path = os.path.join(OUTPUT_FOLDER, "embeddings.pkl")
    with open(emb_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {os.path.basename(emb_path)}\n")
    
    # Step 5: Clustering
    umap_embeds, labels_hdb, kmeans_labels, best_k = perform_clustering(embeddings)
    
    # Step 6: Calculate metrics
    print(f"Clustering Quality Metrics:")
    calculate_metrics(umap_embeds, labels_hdb, "HDBSCAN")
    calculate_metrics(umap_embeds, kmeans_labels, f"KMeans (K={best_k})")
    
    # Step 7: Save results
    save_cluster_results(
        labels_hdb, "HDBSCAN", 
        texts_split, file_names_split, file_to_original
    )
    save_cluster_results(
        kmeans_labels, f"KMeans_K{best_k}", 
        texts_split, file_names_split, file_to_original
    )
    
    # Step 8: Visualization
    create_visualizations(embeddings, labels_hdb, kmeans_labels, best_k)
    
    # Summary
    n_clusters_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
    
    print(f"\n{'#'*70}")
    print(f"# CLUSTERING COMPLETED SUCCESSFULLY")
    print(f"{'#'*70}")
    print(f"\nResults Summary:")
    print(f"   - Input texts: {len(file_names)}")
    print(f"   - Text chunks: {len(texts_split)}")
    print(f"   - HDBSCAN clusters: {n_clusters_hdb}")
    print(f"   - KMeans clusters: {best_k}")
    print(f"\nOutput folder: {os.path.relpath(OUTPUT_FOLDER, PROJECT_ROOT)}")
    print(f"\nGenerated files:")
    print(f"   - embeddings.pkl")
    print(f"   - HDBSCAN_clusters.csv")
    print(f"   - HDBSCAN_representatives.txt")
    print(f"   - HDBSCAN_meta.json")
    print(f"   - KMeans_K{best_k}_clusters.csv")
    print(f"   - KMeans_K{best_k}_representatives.txt")
    print(f"   - KMeans_K{best_k}_meta.json")
    print(f"   - Visualization PNGs\n")

if __name__ == "__main__":
    main()
input("\nPress Enter to exit...")
