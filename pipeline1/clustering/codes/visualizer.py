#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cluster Visualization v2.0
- PCA, t-SNE, UMAP visualizations
- Compatible with new cluster_engine.py output format
- Reads settings from config.yaml
"""

import os
import sys
import pickle
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
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
SCRIPT_DIR = Path(__file__).parent.resolve()      # clustering/codes/
CLUSTERING_DIR = SCRIPT_DIR.parent                 # clustering/
PIPELINE_DIR = CLUSTERING_DIR.parent               # pipeline1/
PROJECT_ROOT = PIPELINE_DIR.parent                 # AAI6610_FALL2025/

# Input/Output paths
OUTPUT_ROOT = CLUSTERING_DIR / "outputs"
CLUSTER_OUTPUT = OUTPUT_ROOT / "cluster_output"
VIS_OUTPUT = OUTPUT_ROOT / "cluster_visualizations"

# Config paths
CONFIG_PATHS = [PIPELINE_DIR / "config.yaml", PROJECT_ROOT / "config.yaml"]

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
VIS_CONFIG = CONFIG.get('visualization', {})

DPI = VIS_CONFIG.get('dpi', 300)
FIGURE_SIZE = tuple(VIS_CONFIG.get('figure_size', [12, 10]))
METHODS = VIS_CONFIG.get('methods', ['umap', 'pca', 'tsne'])

# ============================================================================
# Data Loading
# ============================================================================
def load_embeddings(embeddings_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings from pickle file"""
    logger.info(f"Loading embeddings from {embeddings_path.name}...")
    
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both old format (list) and new format (dict)
    if isinstance(data, dict):
        embeddings = np.array(data.get('embeddings', data))
        filenames = data.get('filenames', [])
    elif isinstance(data, list):
        embeddings = np.array(data)
        filenames = []
    else:
        embeddings = np.array(data)
        filenames = []
    
    logger.info(f"  Shape: {embeddings.shape}")
    logger.info(f"  Filenames: {len(filenames)}")
    
    return embeddings, filenames

def find_cluster_files(cluster_dir: Path) -> List[Path]:
    """Find all cluster result CSV files"""
    files = list(cluster_dir.glob("*_clusters.csv"))
    
    if not files:
        logger.warning("No cluster files found!")
        return []
    
    logger.info(f"Found {len(files)} cluster file(s):")
    for f in files:
        logger.info(f"  - {f.name}")
    
    return files

def load_cluster_labels(csv_path: Path) -> np.ndarray:
    """Load cluster labels from CSV"""
    df = pd.read_csv(csv_path)
    return df['cluster'].to_numpy()

def load_cluster_metadata(cluster_dir: Path, base_name: str) -> Dict[int, str]:
    """Load cluster metadata to get representative filenames"""
    # Try both naming conventions
    meta_files = [
        cluster_dir / f"{base_name}_metadata.json",  # New format
        cluster_dir / f"{base_name}_meta.json",      # Old format
    ]
    
    for meta_path in meta_files:
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                # Extract first file of each cluster as representative
                cluster_reps = {}
                seen_clusters = set()
                
                for item in meta:
                    cluster_id = item.get('cluster', -1)
                    if cluster_id not in seen_clusters and cluster_id != -1:
                        cluster_reps[cluster_id] = item.get('filename', f'Cluster {cluster_id}')
                        seen_clusters.add(cluster_id)
                
                logger.info(f"  Loaded metadata for {len(cluster_reps)} clusters")
                return cluster_reps
                
            except Exception as e:
                logger.debug(f"Error loading metadata: {e}")
    
    return {}

# ============================================================================
# Dimensionality Reduction
# ============================================================================
def reduce_pca(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
    """PCA reduction for preprocessing"""
    target_dim = min(n_components, embeddings.shape[1] - 1, embeddings.shape[0] - 1)
    
    pca = PCA(n_components=target_dim, random_state=42)
    reduced = pca.fit_transform(embeddings)
    
    explained = pca.explained_variance_ratio_.sum()
    logger.info(f"  PCA: {embeddings.shape[1]}D -> {target_dim}D (explained: {explained*100:.1f}%)")
    
    return reduced

def reduce_to_2d(embeddings: np.ndarray, method: str = 'umap') -> np.ndarray:
    """Reduce embeddings to 2D for visualization"""
    n_samples = embeddings.shape[0]
    
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(embeddings)
    
    elif method == 'tsne':
        perplexity = min(30, n_samples - 1)
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            init='pca',
            random_state=42,
            max_iter=1000
        )
        return reducer.fit_transform(embeddings)
    
    elif method == 'umap':
        n_neighbors = min(15, n_samples - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        return reducer.fit_transform(embeddings)
    
    else:
        raise ValueError(f"Unknown method: {method}")

# ============================================================================
# Plotting
# ============================================================================
def plot_clusters(coords: np.ndarray, labels: np.ndarray, title: str,
                  save_path: Path, cluster_reps: Dict[int, str] = None):
    """Create 2D cluster visualization"""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    
    # Color map
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_clusters + 1)))
    
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        
        if label == -1:
            # Noise points
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c='lightgray', alpha=0.3, s=15, label='Noise'
            )
        else:
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[colors[i % 20]], alpha=0.6, s=40,
                label=f'Cluster {label} ({mask.sum()})'
            )
    
    # Annotate cluster centers
    if cluster_reps:
        for cluster_id, filename in cluster_reps.items():
            if cluster_id == -1:
                continue
            
            mask = labels == cluster_id
            if not mask.any():
                continue
            
            # Cluster center
            cx = coords[mask, 0].mean()
            cy = coords[mask, 1].mean()
            
            # Shortened filename
            short_name = filename[:25] + "..." if len(filename) > 25 else filename
            
            ax.annotate(
                f"C{cluster_id}",
                xy=(cx, cy),
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', 
                         edgecolor='black', alpha=0.8)
            )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend (limit to 15 entries)
    handles, legend_labels = ax.get_legend_handles_labels()
    if len(handles) > 15:
        handles, legend_labels = handles[:15], legend_labels[:15]
        legend_labels[-1] = "... (more clusters)"
    
    ax.legend(handles, legend_labels, loc='upper right', fontsize=8,
              framealpha=0.9, title=f"{n_clusters} Clusters")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

def create_summary_plot(embeddings: np.ndarray, labels: np.ndarray,
                        cluster_name: str, save_path: Path,
                        cluster_reps: Dict[int, str] = None):
    """Create a summary plot with all three methods"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Pre-reduce with PCA for speed
    reduced = reduce_pca(embeddings, n_components=50)
    
    methods = ['pca', 'tsne', 'umap']
    titles = ['PCA', 't-SNE', 'UMAP']
    
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_clusters + 1)))
    
    for ax, method, title in zip(axes, methods, titles):
        logger.info(f"  Computing {title}...")
        
        coords = reduce_to_2d(reduced, method)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            
            if label == -1:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c='lightgray', alpha=0.3, s=10)
            else:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=[colors[i % 20]], alpha=0.6, s=25)
        
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{cluster_name} - {n_clusters} Clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

# ============================================================================
# Main Visualization Pipeline
# ============================================================================
def generate_visualizations(embeddings: np.ndarray, cluster_files: List[Path]):
    """Generate visualizations for all cluster results"""
    
    # Pre-reduce embeddings with PCA
    logger.info("\nPre-processing embeddings...")
    reduced = reduce_pca(embeddings, n_components=50)
    
    for cluster_file in cluster_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {cluster_file.name}")
        logger.info(f"{'='*60}")
        
        # Load cluster labels
        labels = load_cluster_labels(cluster_file)
        
        # Verify dimensions match
        if len(labels) != len(embeddings):
            logger.warning(f"Size mismatch: {len(labels)} labels vs {len(embeddings)} embeddings")
            logger.warning("Skipping this file...")
            continue
        
        base_name = cluster_file.stem.replace('_clusters', '')
        
        # Statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        logger.info(f"  Clusters: {n_clusters}, Noise: {n_noise}, Total: {len(labels)}")
        
        # Load metadata
        cluster_reps = load_cluster_metadata(CLUSTER_OUTPUT, base_name)
        
        # Generate individual visualizations
        for method in METHODS:
            logger.info(f"  Generating {method.upper()} visualization...")
            
            coords = reduce_to_2d(reduced, method)
            
            save_path = VIS_OUTPUT / f"{base_name}_{method}.png"
            plot_clusters(
                coords, labels,
                f"{method.upper()} - {base_name} ({n_clusters} clusters)",
                save_path, cluster_reps
            )
            logger.info(f"    Saved: {save_path.name}")
        
        # Generate summary plot
        logger.info("  Generating summary plot...")
        summary_path = VIS_OUTPUT / f"{base_name}_summary.png"
        create_summary_plot(embeddings, labels, base_name, summary_path, cluster_reps)
        logger.info(f"    Saved: {summary_path.name}")

# ============================================================================
# Main
# ============================================================================
def main():
    print(f"\n{'#'*70}")
    print(f"# CLUSTER VISUALIZATION v2.0")
    print(f"# PCA, t-SNE, UMAP Visualizations")
    print(f"{'#'*70}\n")
    
    # Validate paths
    logger.info("Validating paths...")
    logger.info(f"  Cluster output: {CLUSTER_OUTPUT}")
    logger.info(f"  Visualization output: {VIS_OUTPUT}")
    
    if not CLUSTER_OUTPUT.exists():
        logger.error(f"Cluster output folder not found!")
        logger.info("Please run cluster_engine.py first.")
        return
    
    # Find embeddings file
    embeddings_path = CLUSTER_OUTPUT / "embeddings.pkl"
    if not embeddings_path.exists():
        logger.error("embeddings.pkl not found!")
        return
    
    # Clean output folder
    if VIS_OUTPUT.exists():
        shutil.rmtree(VIS_OUTPUT)
    VIS_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    embeddings, filenames = load_embeddings(embeddings_path)
    
    # Find cluster files
    cluster_files = find_cluster_files(CLUSTER_OUTPUT)
    if not cluster_files:
        return
    
    # Generate visualizations
    generate_visualizations(embeddings, cluster_files)
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"# VISUALIZATION COMPLETE")
    print(f"{'#'*70}")
    print(f"\nOutput: {VIS_OUTPUT.relative_to(PIPELINE_DIR)}")
    print(f"\nGenerated files:")
    for f in sorted(VIS_OUTPUT.glob("*.png")):
        print(f"  - {f.name}")
    
    print(f"\nVisualization Guide:")
    print(f"  PCA:   Linear projection, preserves global structure")
    print(f"  t-SNE: Non-linear, emphasizes local neighborhoods")
    print(f"  UMAP:  Balanced, preserves both local and global structure")
    print()

if __name__ == "__main__":
    main()