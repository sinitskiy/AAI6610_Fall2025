#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Semantic Filter - Multi-source Data Support (PDF + TXT) - Relative Path Version
"""

import sys
import io

# Windows GBK encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from typing import List, Dict
import json

# PDF parsing library
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    print("Warning: pypdf not installed. PDF files will be skipped.")
    print("   Install with: pip install pypdf")
    PDF_AVAILABLE = False

# ============================================================================
# Configuration - Using relative paths
# ============================================================================
# Get script directory (clustering/codes/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root directory (AAI6610_WholePipeline/)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Data source root directory (scrapers/outputs/)
DATA_ROOT = os.path.join(PROJECT_ROOT, "scrapers", "outputs")

INPUT_FOLDERS = [
    # Paper sources
    os.path.join(DATA_ROOT, "arxiv_papers"),
    os.path.join(DATA_ROOT, "biorxiv_papers"),
    os.path.join(DATA_ROOT, "openalex_openreview_papers", "openalex"),
    os.path.join(DATA_ROOT, "openalex_openreview_papers", "openreview"),
    os.path.join(DATA_ROOT, "openalex_openreview_papers", "unpaywall"),
    
    # Social media sources
    os.path.join(DATA_ROOT, "linkedin_posts"),
    os.path.join(DATA_ROOT, "reddit_posts"),
    
    # News sources
    os.path.join(DATA_ROOT, "news_articles"),
]

# Output folder (clustering/outputs/)
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "..", "outputs")
OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, "filtered_posts_all_sources")
STATS_FOLDER = os.path.join(OUTPUT_ROOT, "filter_stats_all_sources")

# Ensure output root directory exists
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Semantic filtering parameters
MODEL_NAME = 'all-MiniLM-L6-v2'
THRESHOLD = 30.0  # Adjustable: 20(loose) - 30(balanced) - 40(strict)
BATCH_SIZE = 16

# Uncertainty topic queries
UNCERTAINTY_QUERIES = [
    "uncertainty estimation in deep learning models",
    "Bayesian neural networks for uncertainty quantification", 
    "epistemic and aleatoric uncertainty in machine learning",
    "confidence calibration in neural networks",
    "probabilistic predictions and uncertainty measures",
    "out-of-distribution detection using uncertainty",
    "predictive uncertainty in AI systems",
    "uncertainty-aware deep learning methods",
    "Monte Carlo dropout for uncertainty estimation",
    "ensemble methods for uncertainty quantification"
]

# ============================================================================
# PDF Text Extraction
# ============================================================================
def extract_text_from_pdf(pdf_path: str, max_pages: int = 10) -> str:
    """Extract text from PDF (first few pages)"""
    if not PDF_AVAILABLE:
        return ""
    
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        
        # Only read first max_pages pages
        num_pages = min(len(reader.pages), max_pages)
        
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return "\n".join(text_parts)
    
    except Exception as e:
        print(f"      Error extracting PDF {os.path.basename(pdf_path)}: {e}")
        return ""

# ============================================================================
# Clean Previous Results
# ============================================================================
def clean_previous_results():
    """Clean results from previous run"""
    folders_to_clean = [OUTPUT_FOLDER, STATS_FOLDER]
    
    print(f"\nCleaning previous results...")
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"   Removed old folder: {os.path.basename(folder)}/")
            except Exception as e:
                print(f"   Could not remove {os.path.basename(folder)}/: {e}")
        
        os.makedirs(folder, exist_ok=True)
        print(f"   Created clean folder: {os.path.basename(folder)}/")
    print()

# ============================================================================
# Validate Folder Paths
# ============================================================================
def validate_input_folders():
    """Validate input folders exist"""
    print(f"\nValidating paths...")
    print(f"   Script location: {SCRIPT_DIR}")
    print(f"   Project root:    {PROJECT_ROOT}")
    print(f"   Data root:       {DATA_ROOT}")
    print(f"   Output root:     {OUTPUT_ROOT}\n")
    
    print(f"Checking input folders:")
    
    missing_folders = []
    existing_folders = []
    
    for folder in INPUT_FOLDERS:
        folder_name = os.path.relpath(folder, PROJECT_ROOT)
        if os.path.exists(folder):
            existing_folders.append(folder)
            print(f"   {folder_name}")
        else:
            missing_folders.append(folder)
            print(f"   {folder_name}")
    
    print(f"\n   Total: {len(existing_folders)}/{len(INPUT_FOLDERS)} folders found")
    
    if missing_folders:
        print(f"\nWarning: {len(missing_folders)} folder(s) not found!")
        print(f"   The script will continue with available folders.\n")
    
    return existing_folders

# ============================================================================
# Semantic Filter
# ============================================================================
class UncertaintySemanticFilter:
    """Semantic filter for uncertainty topics"""
    
    def __init__(self, queries: List[str], model_name: str = MODEL_NAME):
        print(f"Initializing Semantic Filter...")
        print(f"   Model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.queries = queries
        self._query_embeddings = None
        
    def _encode_queries(self):
        """Encode queries (execute only once)"""
        if self._query_embeddings is None:
            print(f"Encoding {len(self.queries)} uncertainty queries...")
            self._query_embeddings = self.model.encode(
                self.queries,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            print("   Queries encoded")
        return self._query_embeddings
    
    def compute_scores(self, texts: List[str], batch_size: int = BATCH_SIZE):
        """Compute text relevance scores"""
        print(f"\nComputing semantic scores for {len(texts)} texts...")
        
        query_embeddings = self._encode_queries()
        
        print("   Encoding documents...")
        doc_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print("   Computing similarities...")
        similarities = cosine_similarity(query_embeddings, doc_embeddings)
        
        max_similarity = similarities.max(axis=0)
        avg_similarity = similarities.mean(axis=0)
        
        scores = (max_similarity * 100).clip(0, 100)
        
        print(f"   Score range: {scores.min():.2f} - {scores.max():.2f}")
        print(f"   Mean score: {scores.mean():.2f}")
        
        return {
            'scores': scores,
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity
        }

# ============================================================================
# Text Reading (PDF and TXT Support)
# ============================================================================
def read_text_files(folders: List[str]) -> pd.DataFrame:
    """
    Read text files from multiple folders (supports PDF and TXT)
    
    Returns: DataFrame with columns [filename, text, source, file_type]
    """
    print(f"\nReading files from {len(folders)} folders...")
    
    all_data = []
    
    for folder in folders:
        if not os.path.exists(folder):
            print(f"   Folder not found: {os.path.relpath(folder, PROJECT_ROOT)}, skipping")
            continue
        
        folder_path = Path(folder)
        
        # Get TXT and PDF files
        txt_files = list(folder_path.glob("*.txt"))
        pdf_files = list(folder_path.glob("*.pdf")) if PDF_AVAILABLE else []
        
        total_files = len(txt_files) + len(pdf_files)
        print(f"   {os.path.basename(folder)}: {len(txt_files)} TXT + {len(pdf_files)} PDF = {total_files} files")
        
        # Read TXT files
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Skip empty files
                    if len(content.strip()) < 50:
                        continue
                    
                    all_data.append({
                        'filename': file_path.name,
                        'text': content,
                        'source': os.path.basename(folder),
                        'file_type': 'txt'
                    })
            except Exception as e:
                print(f"      Error reading {file_path.name}: {e}")
        
        # Read PDF files
        for file_path in pdf_files:
            try:
                # Check if corresponding txt file exists (avoid duplication)
                txt_counterpart = file_path.with_suffix('.txt')
                if txt_counterpart.exists():
                    continue  # Skip, already read txt version
                
                content = extract_text_from_pdf(str(file_path))
                
                # Skip failed extraction or too little content
                if len(content.strip()) < 100:
                    continue
                
                all_data.append({
                    'filename': file_path.name,
                    'text': content,
                    'source': os.path.basename(folder),
                    'file_type': 'pdf'
                })
            except Exception as e:
                print(f"      Error processing PDF {file_path.name}: {e}")
    
    df = pd.DataFrame(all_data)
    print(f"\nTotal texts loaded: {len(df)}")
    
    if len(df) > 0:
        print(f"\nSource distribution:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"   - {source}: {count}")
        
        print(f"\nFile type distribution:")
        type_counts = df['file_type'].value_counts()
        for ftype, count in type_counts.items():
            print(f"   - {ftype.upper()}: {count}")
    
    return df

# ============================================================================
# Filter Function
# ============================================================================
def filter_texts(df: pd.DataFrame, threshold: float = THRESHOLD):
    """Filter texts based on semantic scores"""
    print(f"\n{'='*70}")
    print(f"SEMANTIC FILTERING")
    print(f"{'='*70}")
    print(f"Input texts: {len(df)}")
    print(f"Threshold: {threshold}")
    print(f"{'='*70}\n")
    
    filter_model = UncertaintySemanticFilter(UNCERTAINTY_QUERIES)
    
    texts = df['text'].tolist()
    score_results = filter_model.compute_scores(texts)
    
    df['semantic_score'] = score_results['scores']
    df['max_similarity'] = score_results['max_similarity']
    df['avg_similarity'] = score_results['avg_similarity']
    
    filtered_df = df[df['semantic_score'] >= threshold].copy()
    
    print(f"\n{'='*70}")
    print(f"FILTERING RESULTS")
    print(f"{'='*70}")
    print(f"Original:  {len(df):>6} texts")
    print(f"Kept:      {len(filtered_df):>6} texts ({len(filtered_df)/len(df)*100:>5.1f}%)")
    print(f"Removed:   {len(df)-len(filtered_df):>6} texts ({(len(df)-len(filtered_df))/len(df)*100:>5.1f}%)")
    print(f"{'='*70}\n")
    
    if len(df) > 0:
        print("Filtering by source:")
        for source in sorted(df['source'].unique()):
            source_total = len(df[df['source'] == source])
            source_kept = len(filtered_df[filtered_df['source'] == source])
            print(f"  {source:30s}: {source_kept:>4}/{source_total:>4} ({source_kept/source_total*100:>5.1f}%)")
    
    return filtered_df, df

# ============================================================================
# Save Results
# ============================================================================
def save_filtered_texts(filtered_df: pd.DataFrame, output_folder: str):
    """Save filtered texts to unified folder (with source prefix)"""
    import hashlib
    
    rel_path = os.path.relpath(output_folder, PROJECT_ROOT)
    print(f"\nSaving filtered texts to {rel_path}/")
    
    # Create unified output folder
    os.makedirs(output_folder, exist_ok=True)
    
    saved_count = 0
    error_count = 0
    
    for idx, row in filtered_df.iterrows():
        try:
            # Uniformly save as txt format
            if row['file_type'] == 'pdf':
                filename = os.path.splitext(row['filename'])[0] + '.txt'
            else:
                filename = row['filename']
            
            # Add source prefix to filename (avoid conflicts from same-named files from different sources)
            source_prefix = row['source'].replace('_', '-')[:20]  # First 20 chars of source name, replace underscores
            filename_with_source = f"[{source_prefix}]_{filename}"
            
            # Handle overly long filenames
            max_filename_length = 150
            if len(filename_with_source) > max_filename_length:
                name_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
                ext = '.txt'
                base_name = filename_with_source[:135]
                # Remove potentially incomplete word at end
                if ' ' in base_name:
                    base_name = base_name.rsplit(' ', 1)[0]
                filename_with_source = f"{base_name}_{name_hash}{ext}"
            
            # Save directly under output_folder
            output_path = os.path.join(output_folder, filename_with_source)
            
            # Check full path length (Windows limitation)
            if len(output_path) > 250:
                short_name = f"[{source_prefix}]_doc{idx}_{hashlib.md5(row['filename'].encode()).hexdigest()[:12]}.txt"
                output_path = os.path.join(output_folder, short_name)
            
            # Save file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(row['text'])
            
            saved_count += 1
            
            # Show progress (every 100 files)
            if saved_count % 100 == 0:
                print(f"      Progress: {saved_count}/{len(filtered_df)} files saved...")
            
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Only show first 5 errors
                print(f"      Error saving file {idx}: {str(e)[:100]}")
    
    print(f"   Successfully saved: {saved_count}/{len(filtered_df)} files in one folder")
    if error_count > 0:
        print(f"   Failed to save: {error_count} files")






def save_statistics(filtered_df: pd.DataFrame, all_df: pd.DataFrame, stats_folder: str):
    """Save statistics and visualizations"""
    print(f"\nGenerating statistics and visualizations...")
    
    # 1. Save scores CSV
    all_df[['filename', 'source', 'file_type', 'semantic_score', 'max_similarity', 'avg_similarity']].to_csv(
        os.path.join(stats_folder, 'all_scores.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    
    filtered_df[['filename', 'source', 'file_type', 'semantic_score', 'max_similarity', 'avg_similarity']].to_csv(
        os.path.join(stats_folder, 'filtered_scores.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    
    
    
    
    # 2. Save JSON summary
    summary = {
        'total_input': len(all_df),
        'total_filtered': len(filtered_df),
        'filter_rate': len(filtered_df) / len(all_df) * 100 if len(all_df) > 0 else 0,
        'threshold_used': THRESHOLD,
        'score_stats': {
            'all': {
                'mean': float(all_df['semantic_score'].mean()),
                'median': float(all_df['semantic_score'].median()),
                'min': float(all_df['semantic_score'].min()),
                'max': float(all_df['semantic_score'].max())
            },
            'filtered': {
                'mean': float(filtered_df['semantic_score'].mean()) if len(filtered_df) > 0 else 0,
                'median': float(filtered_df['semantic_score'].median()) if len(filtered_df) > 0 else 0,
                'min': float(filtered_df['semantic_score'].min()) if len(filtered_df) > 0 else 0,
                'max': float(filtered_df['semantic_score'].max()) if len(filtered_df) > 0 else 0
            }
        },
        'by_source': {}
    }
    
    for source in all_df['source'].unique():
        source_total = len(all_df[all_df['source'] == source])
        source_kept = len(filtered_df[filtered_df['source'] == source])
        summary['by_source'][source] = {
            'total': source_total,
            'kept': source_kept,
            'rate': source_kept / source_total * 100 if source_total > 0 else 0
        }
    
    with open(os.path.join(stats_folder, 'filter_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 3. Generate visualizations
    generate_visualizations(all_df, filtered_df, stats_folder)
    
    rel_path = os.path.relpath(stats_folder, PROJECT_ROOT)
    print(f"   Statistics saved to {rel_path}/")

# ============================================================================
# Visualizations
# ============================================================================
def generate_visualizations(all_df: pd.DataFrame, filtered_df: pd.DataFrame, stats_folder: str):
    """Generate visualization charts"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Chinese support
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Score distribution histogram
    ax = axes[0, 0]
    ax.hist(all_df['semantic_score'], bins=50, alpha=0.6, color='gray', label='All', edgecolor='black')
    ax.hist(filtered_df['semantic_score'], bins=50, alpha=0.8, color='green', label='Kept', edgecolor='black')
    ax.axvline(THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold={THRESHOLD}')
    ax.set_xlabel('Semantic Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution
    ax = axes[0, 1]
    sorted_scores = np.sort(all_df['semantic_score'])
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    ax.plot(sorted_scores, cumulative, linewidth=2, color='steelblue')
    ax.axvline(THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold={THRESHOLD}')
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Semantic Score', fontsize=12)
    ax.set_ylabel('Cumulative %', fontsize=12)
    ax.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Statistics by source (stacked bar chart)
    ax = axes[0, 2]
    sources = sorted(all_df['source'].unique())
    kept_counts = [len(filtered_df[filtered_df['source'] == s]) for s in sources]
    removed_counts = [len(all_df[all_df['source'] == s]) - k for s, k in zip(sources, kept_counts)]
    
    x = np.arange(len(sources))
    width = 0.6
    
    ax.bar(x, kept_counts, width, label='Kept', color='lightgreen')
    ax.bar(x, removed_counts, width, bottom=kept_counts, label='Removed', color='lightcoral')
    
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Filtering by Source', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=45, ha='right', fontsize=9)
    ax.legend()
    
    # 4. Box plot comparison
    ax = axes[1, 0]
    kept_scores = filtered_df['semantic_score'].values
    removed_scores = all_df[all_df['semantic_score'] < THRESHOLD]['semantic_score'].values
    
    bp = ax.boxplot([kept_scores, removed_scores], labels=['Kept', 'Removed'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Semantic Score', fontsize=12)
    ax.set_title('Score Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Average score by source
    ax = axes[1, 1]
    source_means = []
    for source in sources:
        mean_score = all_df[all_df['source'] == source]['semantic_score'].mean()
        source_means.append(mean_score)
    
    colors = ['lightgreen' if m >= THRESHOLD else 'lightcoral' for m in source_means]
    ax.barh(sources, source_means, color=colors)
    ax.axvline(THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold={THRESHOLD}')
    ax.set_xlabel('Mean Semantic Score', fontsize=12)
    ax.set_title('Average Score by Source', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # 6. Statistics table
    ax = axes[1, 2]
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Value'],
        ['', ''],
        ['Total Input', f"{len(all_df)}"],
        ['Kept', f"{len(filtered_df)} ({len(filtered_df)/len(all_df)*100:.1f}%)"],
        ['Removed', f"{len(all_df)-len(filtered_df)} ({(len(all_df)-len(filtered_df))/len(all_df)*100:.1f}%)"],
        ['', ''],
        ['Score (Kept)', ''],
        ['  Mean', f"{filtered_df['semantic_score'].mean():.2f}" if len(filtered_df) > 0 else "N/A"],
        ['  Median', f"{filtered_df['semantic_score'].median():.2f}" if len(filtered_df) > 0 else "N/A"],
        ['  Min', f"{filtered_df['semantic_score'].min():.2f}" if len(filtered_df) > 0 else "N/A"],
        ['  Max', f"{filtered_df['semantic_score'].max():.2f}" if len(filtered_df) > 0 else "N/A"],
        ['', ''],
        ['Sources', f"{len(sources)}"],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(stats_folder, 'filtering_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Visualization saved: filtering_analysis.png")

# ============================================================================
# Display Samples
# ============================================================================
def print_sample_texts(filtered_df: pd.DataFrame, n_samples: int = 5):
    """Print high-scoring samples"""
    if len(filtered_df) == 0:
        print("\nNo texts passed the filter!")
        return
    
    print(f"\n{'='*70}")
    print(f"TOP {min(n_samples, len(filtered_df))} MOST RELEVANT TEXTS")
    print(f"{'='*70}\n")
    
    top_samples = filtered_df.nlargest(min(n_samples, len(filtered_df)), 'semantic_score')
    
    for idx, (i, row) in enumerate(top_samples.iterrows(), 1):
        print(f"[{idx}] Score: {row['semantic_score']:.2f} | Source: {row['source']} | File: {row['filename']}")
        text_preview = row['text'][:300].replace('\n', ' ')
        print(f"    {text_preview}...")
        print()

# ============================================================================
# Main Function
# ============================================================================
def main():
    print(f"\n{'#'*70}")
    print(f"# PIPELINE 2: SEMANTIC FILTERING FOR UNCERTAINTY TOPICS")
    print(f"# MULTI-SOURCE VERSION (PDF + TXT)")
    print(f"{'#'*70}\n")
    
    # Step 0: Validate paths
    existing_folders = validate_input_folders()
    
    if not existing_folders:
        print("No valid input folders found! Please check your directory structure.")
        return
    
    # Step 1: Clean previous results
    clean_previous_results()
    
    # Step 2: Read texts
    df = read_text_files(existing_folders)
    
    if len(df) == 0:
        print("No texts found! Please check input folders.")
        return
    
    # Step 3: Semantic filtering
    filtered_df, all_df = filter_texts(df, threshold=THRESHOLD)
    
    # Step 4: Save filtered texts
    save_filtered_texts(filtered_df, OUTPUT_FOLDER)
    
    # Step 5: Save statistics and visualizations
    save_statistics(filtered_df, all_df, STATS_FOLDER)
    
    # Step 6: Display samples
    print_sample_texts(filtered_df, n_samples=5)
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"# FILTERING COMPLETED SUCCESSFULLY")
    print(f"{'#'*70}")
    print(f"\nOutput folders:")
    print(f"   - Filtered texts: {os.path.relpath(OUTPUT_FOLDER, PROJECT_ROOT)}")
    print(f"   - Statistics:     {os.path.relpath(STATS_FOLDER, PROJECT_ROOT)}")
    print(f"\nNext step: Run clustering analysis on filtered data\n")

if __name__ == "__main__":
    main()
