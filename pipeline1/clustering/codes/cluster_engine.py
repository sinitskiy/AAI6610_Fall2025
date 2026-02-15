#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clustering Engine v2.0 - Open Source Version
Uses sentence-transformers instead of OpenAI
"""

import os
import sys
import re
import pickle
import json
import shutil
import logging
from pathlib import Path
from html import unescape
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan
import umap

# Sentence Transformers (replaces OpenAI)
from sentence_transformers import SentenceTransformer

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
SCRIPT_DIR = Path(__file__).parent.resolve()  # clustering/codes/
CLUSTERING_DIR = SCRIPT_DIR.parent             # clustering/
PIPELINE_DIR = CLUSTERING_DIR.parent           # pipeline1/
PROJECT_ROOT = PIPELINE_DIR.parent             # AAI6610_FALL2025/

# Config file path
CONFIG_PATH = PIPELINE_DIR / "config.yaml"

# ============================================================================
# Load Configuration
# ============================================================================
def load_config() -> dict:
    """Load configuration from config.yaml"""
    if not CONFIG_PATH.exists():
        logger.warning(f"Config file not found: {CONFIG_PATH}")
        logger.info("Using default configuration...")
        return get_default_config()
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {CONFIG_PATH.name}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return get_default_config()

def get_default_config() -> dict:
    """Default configuration if config.yaml is missing"""
    return {
        'clustering': {
            'model': 'all-mpnet-base-v2',  # Better than MiniLM for clustering
            'max_k': 20,
            'min_k': 5,
            'min_cluster_size': 10,
            'min_samples': 3,
            'n_components': 50,
            'n_neighbors': 15,
            'min_dist': 0.1,
            'max_words_per_text': 500,
            'top_keywords': 10,
            'batch_size': 32,
            'use_gpu': False,
        },
        'output': {
            'base_dir': 'outputs',
        }
    }

# ============================================================================
# Initialize NLTK
# ============================================================================
def init_nltk():
    """Download required NLTK data"""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading NLTK data...")
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

init_nltk()

# Global NLP tools
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
CUSTOM_STOPWORDS = {
    '2025', '2024', '10', 'author', 'url', 'unknown', 'date',
    'ai', 'data', 'learning', 'uncertainty', 'model', 'models',
    'linkedin', 'reddit', 'post', 'article', 'paper', 'http', 'https'
}

# ============================================================================
# Embedding Engine (Replaces OpenAI)
# ============================================================================
class EmbeddingEngine:
    """Sentence-Transformers based embedding engine"""
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2', use_gpu: bool = False):
        self.model_name = model_name
        self.device = 'cuda' if use_gpu else 'cpu'
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts: List[str], batch_size: int = 32, 
               show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

# ============================================================================
# Text Preprocessing
# ============================================================================
def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    # Remove URLs and HTML
    text = re.sub(r"http\S+|<.*?>", " ", text)
    text = unescape(text)
    
    # Remove LinkedIn/Reddit UI elements
    ui_patterns = [
        r"Jump to main content", r"LinkedIn.*?content", r"Member",
        r"Join now", r"Log in", r"Download APP"
    ]
    for pattern in ui_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    
    # Keep only alphanumeric
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Lemmatization and stopword removal
    all_stopwords = STOP_WORDS.union(CUSTOM_STOPWORDS)
    words = [
        LEMMATIZER.lemmatize(w.lower())
        for w in text.split()
        if w.lower() not in all_stopwords and len(w) > 2
    ]
    
    return ' '.join(words)

def extract_title_and_preview(text: str, max_preview: int = 500) -> Tuple[str, str]:
    """Extract title and preview from original text"""
    if not text or not text.strip():
        return "No Title", "No content available."
    
    lines = text.strip().split('\n')
    
    # Patterns to skip
    skip_patterns = [
        'Jump to main', 'LinkedIn', 'Cookie', 'Source:', 'URL:',
        'Date:', 'Author:', 'http://', 'https://', 'Published:'
    ]
    
    content_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 15:
            continue
        if any(p.lower() in line.lower() for p in skip_patterns):
            continue
        content_lines.append(line)
    
    if not content_lines:
        return "No Title", "No content available."
    
    # Title: first meaningful line
    title = content_lines[0][:150]
    if len(content_lines[0]) > 150:
        title += "..."
    
    # Preview: first few lines
    preview = ' '.join(content_lines[:5])[:max_preview]
    if len(' '.join(content_lines[:5])) > max_preview:
        preview += "..."
    
    return title, preview

# ============================================================================
# Text Chunking
# ============================================================================
def chunk_texts(texts: List[str], file_names: List[str], 
                max_words: int = 500) -> Tuple[List[str], List[str]]:
    """Split long texts into chunks"""
    chunked_texts = []
    chunked_names = []
    
    for text, fname in zip(texts, file_names):
        words = text.split()
        if len(words) <= max_words:
            chunked_texts.append(text)
            chunked_names.append(fname)
        else:
            # Split into chunks
            for i in range(0, len(words), max_words):
                chunk = ' '.join(words[i:i + max_words])
                chunked_texts.append(chunk)
                chunked_names.append(fname)
    
    logger.info(f"Chunked {len(file_names)} files into {len(chunked_texts)} chunks")
    return chunked_texts, chunked_names

# ============================================================================
# Clustering
# ============================================================================
class ClusteringEngine:
    """UMAP + HDBSCAN/KMeans clustering"""
    
    def __init__(self, config: dict):
        self.config = config.get('clustering', {})
        self.n_components = self.config.get('n_components', 50)
        self.n_neighbors = self.config.get('n_neighbors', 15)
        self.min_dist = self.config.get('min_dist', 0.1)
        self.min_cluster_size = self.config.get('min_cluster_size', 10)
        self.min_samples = self.config.get('min_samples', 3)
        self.min_k = self.config.get('min_k', 5)
        self.max_k = self.config.get('max_k', 20)
    
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """UMAP dimensionality reduction"""
        target_dim = min(self.n_components, embeddings.shape[1] - 1, embeddings.shape[0] - 2)
        
        logger.info(f"UMAP: {embeddings.shape[1]}D -> {target_dim}D")
        
        reducer = umap.UMAP(
            n_neighbors=min(self.n_neighbors, embeddings.shape[0] - 1),
            min_dist=self.min_dist,
            n_components=target_dim,
            metric='cosine',
            random_state=42,
            verbose=False
        )
        
        return reducer.fit_transform(embeddings)
    
    def cluster_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """HDBSCAN clustering"""
        logger.info("Running HDBSCAN clustering...")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        
        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
        return labels
    
    def cluster_kmeans(self, embeddings: np.ndarray) -> Tuple[np.ndarray, int]:
        """KMeans with automatic K selection"""
        logger.info(f"Finding optimal K in range [{self.min_k}, {self.max_k}]...")
        
        # Adjust K range based on data size
        max_k = min(self.max_k, embeddings.shape[0] - 1)
        min_k = min(self.min_k, max_k)
        
        best_score = -1
        best_labels = None
        best_k = min_k
        
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            if len(set(labels)) > 1:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_k = k
        
        logger.info(f"KMeans: Best K={best_k}, Silhouette={best_score:.3f}")
        return best_labels, best_k

# ============================================================================
# Cluster Summarization (Extractive - No OpenAI needed)
# ============================================================================
def generate_extractive_summary(texts: List[str], keywords: List[str], 
                                n_sentences: int = 3) -> str:
    """Generate extractive summary without LLM"""
    if not texts:
        return "No content available for summary."
    
    # Combine texts
    all_text = ' '.join(texts[:5])  # Use first 5 documents
    
    # Simple sentence extraction
    sentences = re.split(r'[.!?]+', all_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    
    if not sentences:
        return f"Cluster about: {', '.join(keywords[:5])}"
    
    # Score sentences by keyword overlap
    keyword_set = set(k.lower() for k in keywords)
    scored = []
    
    for sent in sentences[:20]:  # Check first 20 sentences
        words = set(sent.lower().split())
        overlap = len(words & keyword_set)
        scored.append((overlap, sent))
    
    # Get top sentences
    scored.sort(reverse=True)
    top_sentences = [s for _, s in scored[:n_sentences]]
    
    if top_sentences:
        return ' '.join(top_sentences)
    else:
        return f"Topics include: {', '.join(keywords[:5])}"

# ============================================================================
# Keyword Extraction
# ============================================================================
def extract_keywords(texts: List[str], top_n: int = 10) -> List[str]:
    """Extract top keywords using TF-IDF"""
    if not texts:
        return []
    
    try:
        all_stopwords = list(STOP_WORDS.union(CUSTOM_STOPWORDS))
        
        vectorizer = TfidfVectorizer(
            stop_words=all_stopwords,
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum TF-IDF scores across documents
        scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        
        return [feature_names[i] for i in top_indices]
    
    except Exception as e:
        logger.warning(f"Keyword extraction failed: {e}")
        return []

# ============================================================================
# Metrics Calculation
# ============================================================================
def calculate_metrics(embeddings: np.ndarray, labels: np.ndarray, 
                      name: str) -> Dict[str, float]:
    """Calculate clustering quality metrics"""
    mask = labels != -1
    
    if mask.sum() <= 1 or len(set(labels[mask])) <= 1:
        return {'silhouette': -1, 'calinski_harabasz': -1, 'davies_bouldin': -1}
    
    metrics = {
        'silhouette': silhouette_score(embeddings[mask], labels[mask]),
        'calinski_harabasz': calinski_harabasz_score(embeddings[mask], labels[mask]),
        'davies_bouldin': davies_bouldin_score(embeddings[mask], labels[mask])
    }
    
    logger.info(f"{name} Metrics:")
    logger.info(f"  Silhouette: {metrics['silhouette']:.3f}")
    logger.info(f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}")
    logger.info(f"  Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
    
    return metrics

# ============================================================================
# Results Saving
# ============================================================================
def save_results(labels: np.ndarray, method_name: str, texts_clean: List[str],
                 texts_original: Dict[str, str], file_names: List[str],
                 output_dir: Path, top_keywords: int = 10):
    """Save clustering results"""
    logger.info(f"Saving {method_name} results...")
    
    # 1. Cluster assignments CSV
    df = pd.DataFrame({'filename': file_names, 'cluster': labels})
    df.to_csv(output_dir / f"{method_name}_clusters.csv", index=False, encoding='utf-8-sig')
    
    # 2. Cluster representatives and summaries
    representatives = []
    metadata = []
    
    unique_labels = sorted(set(labels))
    
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue
        
        # Get cluster members
        indices = [i for i, l in enumerate(labels) if l == cluster_id]
        cluster_texts = [texts_clean[i] for i in indices]
        cluster_files = [file_names[i] for i in indices]
        
        # Keywords
        keywords = extract_keywords(cluster_texts, top_keywords)
        
        # Summary (extractive)
        summary = generate_extractive_summary(cluster_texts, keywords)
        
        # Representative document
        rep_file = cluster_files[0]
        rep_original = texts_original.get(rep_file, "")
        title, preview = extract_title_and_preview(rep_original)
        
        # Format representative info
        rep_text = (
            f"{'='*70}\n"
            f"Cluster {cluster_id} | Size: {len(indices)} documents\n"
            f"{'='*70}\n"
            f"Keywords: {', '.join(keywords)}\n\n"
            f"Summary: {summary}\n\n"
            f"Representative: {rep_file}\n"
            f"Title: {title}\n\n"
            f"Preview:\n{preview}\n\n"
        )
        representatives.append(rep_text)
        
        # Metadata for each document
        for idx in indices:
            metadata.append({
                'filename': file_names[idx],
                'cluster': int(cluster_id),
                'keywords': ', '.join(keywords),
                'text_preview': texts_clean[idx][:500]
            })
    
    # Save representatives
    with open(output_dir / f"{method_name}_representatives.txt", 'w', encoding='utf-8') as f:
        f.writelines(representatives)
    
    # Save metadata JSON
    with open(output_dir / f"{method_name}_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  Saved {len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters")

# ============================================================================
# Visualization
# ============================================================================
def create_visualizations(embeddings: np.ndarray, labels_hdb: np.ndarray,
                          labels_km: np.ndarray, best_k: int, output_dir: Path):
    """Create clustering visualizations"""
    logger.info("Creating visualizations...")
    
    # UMAP to 2D for visualization
    reducer = umap.UMAP(
        n_neighbors=min(15, embeddings.shape[0] - 1),
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42
    )
    coords_2d = reducer.fit_transform(embeddings)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # HDBSCAN plot
    scatter1 = axes[0].scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=labels_hdb, cmap='tab20', s=30, alpha=0.7
    )
    axes[0].set_title('HDBSCAN Clustering', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # KMeans plot
    scatter2 = axes[1].scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=labels_km, cmap='tab20', s=30, alpha=0.7
    )
    axes[1].set_title(f'KMeans Clustering (K={best_k})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clustering_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("  Saved clustering_visualization.png")

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    print(f"\n{'#'*70}")
    print(f"# CLUSTERING ENGINE v2.0 (Open Source)")
    print(f"# Using: sentence-transformers")
    print(f"{'#'*70}\n")
    
    # Load configuration
    config = load_config()
    clustering_config = config.get('clustering', {})
    
    # Setup paths
    input_dir = CLUSTERING_DIR / "outputs" / "filtered_posts_all_sources"
    output_dir = CLUSTERING_DIR / "outputs" / "cluster_output"
    
    # Validate input
    if not input_dir.exists():
        logger.error(f"Input folder not found: {input_dir}")
        logger.info("Please run semantic_filter.py first!")
        return
    
    # Clean output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input:  {input_dir.relative_to(PROJECT_ROOT)}")
    logger.info(f"Output: {output_dir.relative_to(PROJECT_ROOT)}")
    
    # Read texts
    logger.info("\nReading input files...")
    txt_files = list(input_dir.glob("*.txt"))
    
    if not txt_files:
        logger.error("No .txt files found in input folder!")
        return
    
    texts_original = {}
    texts_clean = []
    file_names = []
    
    for fpath in txt_files:
        try:
            content = fpath.read_text(encoding='utf-8', errors='ignore')
            if len(content.strip()) < 50:
                continue
            
            texts_original[fpath.name] = content
            texts_clean.append(clean_text(content))
            file_names.append(fpath.name)
        except Exception as e:
            logger.warning(f"Error reading {fpath.name}: {e}")
    
    logger.info(f"Loaded {len(file_names)} valid texts")
    
    if len(file_names) < 10:
        logger.error("Not enough texts for clustering (need at least 10)")
        return
    
    # Chunk texts
    max_words = clustering_config.get('max_words_per_text', 500)
    texts_chunked, names_chunked = chunk_texts(texts_clean, file_names, max_words)
    
    # Generate embeddings
    model_name = clustering_config.get('model', 'all-mpnet-base-v2')
    batch_size = clustering_config.get('batch_size', 32)
    use_gpu = clustering_config.get('use_gpu', False)
    
    embedder = EmbeddingEngine(model_name, use_gpu)
    embeddings = embedder.encode(texts_chunked, batch_size=batch_size)
    
    # Save embeddings
    with open(output_dir / 'embeddings.pkl', 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'filenames': names_chunked}, f)
    logger.info("Saved embeddings.pkl")
    
    # Clustering
    cluster_engine = ClusteringEngine(config)
    
    # Dimensionality reduction
    reduced = cluster_engine.reduce_dimensions(embeddings)
    
    # HDBSCAN
    labels_hdb = cluster_engine.cluster_hdbscan(reduced)
    
    # KMeans
    labels_km, best_k = cluster_engine.cluster_kmeans(reduced)
    
    # Metrics
    print("\n" + "="*50)
    calculate_metrics(reduced, labels_hdb, "HDBSCAN")
    calculate_metrics(reduced, labels_km, f"KMeans(K={best_k})")
    print("="*50 + "\n")
    
    # Save results
    top_kw = clustering_config.get('top_keywords', 10)
    save_results(labels_hdb, "HDBSCAN", texts_chunked, texts_original, 
                 names_chunked, output_dir, top_kw)
    save_results(labels_km, f"KMeans_K{best_k}", texts_chunked, texts_original,
                 names_chunked, output_dir, top_kw)
    
    # Visualizations
    create_visualizations(embeddings, labels_hdb, labels_km, best_k, output_dir)
    
    # Summary
    n_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
    
    print(f"\n{'#'*70}")
    print(f"# CLUSTERING COMPLETE")
    print(f"{'#'*70}")
    print(f"\nResults:")
    print(f"  Input texts:      {len(file_names)}")
    print(f"  Text chunks:      {len(texts_chunked)}")
    print(f"  HDBSCAN clusters: {n_hdb}")
    print(f"  KMeans clusters:  {best_k}")
    print(f"\nOutput: {output_dir.relative_to(PROJECT_ROOT)}")
    print(f"\nModel used: {model_name} ({embedder.embedding_dim}D embeddings)")

if __name__ == "__main__":
    main()