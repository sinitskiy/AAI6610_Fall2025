#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cluster Visualization - Generate PCA, t-SNE, UMAP multi-dimensional visualizations
"""

import sys
import io

# Windows GBK encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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

# ============================================================================
# Configuration - Using relative paths
# ============================================================================
# Get script directory (clustering/codes/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root directory (AAI6610_WholePipeline/)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Input/output paths (clustering/outputs/)
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "..", "outputs")
CLUSTER_OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "cluster_output")
VIS_FOLDER = os.path.join(OUTPUT_ROOT, "cluster_visualizations")

# Embeddings file path
EMBEDDINGS_FILE = os.path.join(CLUSTER_OUTPUT_FOLDER, "embeddings.pkl")

# ============================================================================
# Validate Paths
# ============================================================================
def validate_paths():
    """Validate input paths exist"""
    print(f"\nValidating paths...")
    print(f"   Script location: {SCRIPT_DIR}")
    print(f"   Project root:    {PROJECT_ROOT}")
    print(f"   Input folder:    {os.path.relpath(CLUSTER_OUTPUT_FOLDER, PROJECT_ROOT)}")
    print(f"   Output folder:   {os.path.relpath(VIS_FOLDER, PROJECT_ROOT)}\n")
    
    if not os.path.exists(CLUSTER_OUTPUT_FOLDER):
        print(f"Error: Cluster output folder not found!")
        print(f"   Expected: {CLUSTER_OUTPUT_FOLDER}")
        print(f"   Please run cluster_engine.py first!")
        return False
    
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Error: Embeddings file not found!")
        print(f"   Expected: {EMBEDDINGS_FILE}")
        return False
    
    print(f"All input paths validated\n")
    return True

# ============================================================================
# Clean Old Output
# ============================================================================
def clean_vis_folder():
    """Clean old visualization folder"""
    if os.path.exists(VIS_FOLDER):
        print(f"Cleaning old visualization folder...")
        shutil.rmtree(VIS_FOLDER)
    
    os.makedirs(VIS_FOLDER, exist_ok=True)
    print(f"   Created clean folder: {os.path.basename(VIS_FOLDER)}/\n")

# ============================================================================
# Load Data
# ============================================================================
def load_embeddings():
    """Load embeddings"""
    print(f"Loading embeddings from {os.path.basename(EMBEDDINGS_FILE)}...")
    
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
    
    print(f"   Loaded {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {len(embeddings[0])}\n")
    
    return embeddings

def find_cluster_files():
    """Find all cluster result files"""
    cluster_files = [
        f for f in os.listdir(CLUSTER_OUTPUT_FOLDER) 
        if f.endswith("_clusters.csv")
    ]
    
    if not cluster_files:
        print(f"No cluster files found in {CLUSTER_OUTPUT_FOLDER}")
        return []
    
    print(f"Found {len(cluster_files)} cluster file(s):")
    for f in cluster_files:
        print(f"   - {f}")
    print()
    
    return cluster_files

# ============================================================================
# Dimensionality Reduction
# ============================================================================
def reduce_dimensions(embeddings):
    """Use PCA to reduce to 50 dimensions (unified dimension, speed up subsequent processing)"""
    print(f"Reducing dimensions with PCA...")
    print(f"   {len(embeddings[0])}D -> 50D")
    
    pca50 = PCA(n_components=50, random_state=42)
    reduced_embeddings = pca50.fit_transform(embeddings)
    
    explained_var = pca50.explained_variance_ratio_.sum()
    print(f"   Explained variance: {explained_var*100:.2f}%\n")
    
    return reduced_embeddings

# ============================================================================
# Plotting Functions
# ============================================================================
def plot_2d(coords, labels, title, save_path, cluster_reps=None):
    """Generate 2D visualization plot"""
    plt.figure(figsize=(12, 10))
    
    # Draw scatter plot
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Noise points in gray
            mask = labels == label
            plt.scatter(
                coords[mask, 0], 
                coords[mask, 1], 
                c='gray', 
                alpha=0.3, 
                s=20, 
                label='Noise'
            )
        else:
            mask = labels == label
            plt.scatter(
                coords[mask, 0], 
                coords[mask, 1], 
                c=[colors[i]], 
                alpha=0.6, 
                s=40, 
                label=f'Cluster {label}'
            )
    
    # Annotate cluster centers and representative filenames
    if cluster_reps:
        for c, fname in cluster_reps.items():
            if c == -1:
                continue
            
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            
            x_mean = np.mean(coords[idx, 0])
            y_mean = np.mean(coords[idx, 1])
            
            # Display cluster number and partial filename
            short_name = fname[:30] + "..." if len(fname) > 30 else fname
            
            plt.annotate(
                f"C{c}\n{short_name}",
                xy=(x_mean, y_mean),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                ha='left'
            )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Legend (only show first 10 clusters to avoid crowding)
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    if len(handles) > 11:  # 10 clusters + noise
        plt.legend(handles[:11], labels_legend[:11], 
                  loc='upper right', fontsize=9, title="Clusters")
    else:
        plt.legend(loc='upper right', fontsize=9, title="Clusters")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Generate Multiple Visualizations
# ============================================================================
def create_visualizations(embeddings, reduced_embeddings, cluster_files):
    """Generate PCA/t-SNE/UMAP visualizations for each clustering result"""
    
    for cluster_file in cluster_files:
        print(f"\n{'='*60}")
        print(f"Processing: {cluster_file}")
        print(f"{'='*60}\n")
        
        # Read cluster labels
        cluster_path = os.path.join(CLUSTER_OUTPUT_FOLDER, cluster_file)
        df = pd.read_csv(cluster_path)
        labels = df['cluster'].to_numpy()
        
        # Verify length match
        if len(labels) != len(embeddings):
            print(f"Warning: Label count mismatch!")
            print(f"   Labels: {len(labels)}, Embeddings: {len(embeddings)}")
            print(f"   Skipping this file...\n")
            continue
        
        base_name = os.path.splitext(cluster_file)[0]
        
        # Statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum() if -1 in labels else 0
        print(f"Cluster statistics:")
        print(f"   Clusters: {n_clusters}")
        print(f"   Noise points: {n_noise}")
        print(f"   Total points: {len(labels)}\n")
        
        # Load metadata (get representative filenames)
        meta_file = os.path.join(CLUSTER_OUTPUT_FOLDER, f"{base_name}_meta.json")
        cluster_reps = {}
        
        if os.path.exists(meta_file):
            print(f"Loading metadata from {base_name}_meta.json...")
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            # Extract first file of each cluster as representative
            for c in set(labels):
                if c == -1:
                    continue
                matching = [m['filename'] for m in meta if m['cluster'] == c]
                if matching:
                    cluster_reps[c] = matching[0]
            
            print(f"   Found representatives for {len(cluster_reps)} clusters\n")
        
        # 1. PCA 2D
        print(f"   Generating PCA 2D visualization...")
        pca2 = PCA(n_components=2, random_state=42)
        pca_2d = pca2.fit_transform(reduced_embeddings)
        pca_path = os.path.join(VIS_FOLDER, f"{base_name}_pca2d.png")
        plot_2d(pca_2d, labels, f"PCA 2D - {base_name}", pca_path, cluster_reps)
        print(f"      Saved: {base_name}_pca2d.png")
        
        # 2. t-SNE
        print(f"   Generating t-SNE visualization (this may take a while)...")
        tsne = TSNE(
            n_components=2, 
            perplexity=30, 
            learning_rate=200, 
            random_state=42,
            max_iter=1000
        )
        tsne_2d = tsne.fit_transform(reduced_embeddings)
        tsne_path = os.path.join(VIS_FOLDER, f"{base_name}_tsne.png")
        plot_2d(tsne_2d, labels, f"t-SNE - {base_name}", tsne_path, cluster_reps)
        print(f"      Saved: {base_name}_tsne.png")
        
        # 3. UMAP
        print(f"   Generating UMAP visualization...")
        reducer = umap.UMAP(
            n_neighbors=15, 
            min_dist=0.1, 
            random_state=42,
            n_components=2
        )
        umap_2d = reducer.fit_transform(reduced_embeddings)
        umap_path = os.path.join(VIS_FOLDER, f"{base_name}_umap.png")
        plot_2d(umap_2d, labels, f"UMAP - {base_name}", umap_path, cluster_reps)
        print(f"      Saved: {base_name}_umap.png")
        
        print(f"\nAll visualizations saved for {cluster_file}")

# ============================================================================
# Main Function
# ============================================================================
def main():
    print(f"\n{'#'*70}")
    print(f"# MULTI-DIMENSIONAL VISUALIZATION")
    print(f"# PCA, t-SNE, UMAP Cluster Visualizations")
    print(f"{'#'*70}\n")
    
    # Step 0: Validate paths
    if not validate_paths():
        return
    
    # Step 1: Clean old files
    clean_vis_folder()
    
    # Step 2: Load embeddings
    embeddings = load_embeddings()
    
    # Step 3: Find cluster files
    cluster_files = find_cluster_files()
    if not cluster_files:
        return
    
    # Step 4: PCA dimension reduction to 50D
    reduced_embeddings = reduce_dimensions(embeddings)
    
    # Step 5: Generate visualizations
    create_visualizations(embeddings, reduced_embeddings, cluster_files)
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"# VISUALIZATION COMPLETED SUCCESSFULLY")
    print(f"{'#'*70}")
    print(f"\nOutput folder: {os.path.relpath(VIS_FOLDER, PROJECT_ROOT)}")
    print(f"\nGenerated visualizations:")
    
    vis_files = sorted(os.listdir(VIS_FOLDER))
    for f in vis_files:
        print(f"   - {f}")
    
    print(f"\nTip: Compare PCA, t-SNE, and UMAP to understand cluster structure")
    print(f"   - PCA: Linear, preserves global structure")
    print(f"   - t-SNE: Non-linear, emphasizes local structure")
    print(f"   - UMAP: Balanced, preserves both local and global structure\n")

if __name__ == "__main__":
    main()
