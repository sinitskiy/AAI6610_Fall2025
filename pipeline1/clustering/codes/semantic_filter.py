#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Semantic Filter v2.0 - Multi-source Data Support
- Reads settings from config.yaml
- Consistent paths with pipeline1/ structure
- Better PDF handling and statistics
"""

import os
import sys
import shutil
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# PDF support
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: pypdf not installed. PDF files will be skipped.")

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

# Config paths (check multiple locations)
CONFIG_PATHS = [
    PIPELINE_DIR / "config.yaml",
    PROJECT_ROOT / "config.yaml",
]

# Input: Scraper outputs
SCRAPERS_OUTPUT = PIPELINE_DIR / "scrapers" / "outputs"

INPUT_FOLDERS = [
    # Academic papers
    SCRAPERS_OUTPUT / "arxiv_papers",
    SCRAPERS_OUTPUT / "biorxiv_papers",
    SCRAPERS_OUTPUT / "openalex_openreview_papers" / "openalex",
    SCRAPERS_OUTPUT / "openalex_openreview_papers" / "openreview",
    SCRAPERS_OUTPUT / "openalex_openreview_papers" / "unpaywall",
    # Social & News
    SCRAPERS_OUTPUT / "reddit_posts",
    SCRAPERS_OUTPUT / "linkedin_posts",
    SCRAPERS_OUTPUT / "news_articles",
]

# Output directories
OUTPUT_ROOT = CLUSTERING_DIR / "outputs"
OUTPUT_FOLDER = OUTPUT_ROOT / "filtered_posts_all_sources"
STATS_FOLDER = OUTPUT_ROOT / "filter_stats"

# ============================================================================
# Load Configuration
# ============================================================================
def load_config() -> dict:
    """Load configuration from config.yaml"""
    for config_path in CONFIG_PATHS:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded config from {config_path.name}")
                return config
            except Exception as e:
                logger.warning(f"Error loading {config_path}: {e}")
    
    logger.warning("No config.yaml found, using defaults")
    return get_default_config()

def get_default_config() -> dict:
    """Default configuration"""
    return {
        'semantic_filter': {
            'model': 'all-MiniLM-L6-v2',
            'threshold': 30.0,
            'batch_size': 32,
            'queries': [
                "uncertainty estimation in deep learning models",
                "Bayesian neural networks for uncertainty quantification",
                "epistemic and aleatoric uncertainty in machine learning",
                "confidence calibration in neural networks",
                "probabilistic predictions and uncertainty measures",
                "out-of-distribution detection using uncertainty",
                "predictive uncertainty in AI systems",
                "Monte Carlo dropout for uncertainty estimation",
                "ensemble methods for uncertainty quantification",
                "conformal prediction for uncertainty",
            ]
        }
    }

# Load config
CONFIG = load_config()
FILTER_CONFIG = CONFIG.get('semantic_filter', get_default_config()['semantic_filter'])

MODEL_NAME = FILTER_CONFIG.get('model', 'all-MiniLM-L6-v2')
THRESHOLD = FILTER_CONFIG.get('threshold', 30.0)
BATCH_SIZE = FILTER_CONFIG.get('batch_size', 32)
QUERIES = FILTER_CONFIG.get('queries', get_default_config()['semantic_filter']['queries'])

# ============================================================================
# PDF Text Extraction
# ============================================================================
def extract_text_from_pdf(pdf_path: Path, max_pages: int = 10) -> str:
    """Extract text from PDF file"""
    if not PDF_AVAILABLE:
        return ""
    
    try:
        reader = PdfReader(str(pdf_path))
        text_parts = []
        
        num_pages = min(len(reader.pages), max_pages)
        
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return "\n".join(text_parts)
    
    except Exception as e:
        logger.debug(f"PDF extraction error ({pdf_path.name}): {e}")
        return ""

# ============================================================================
# Semantic Filter Class
# ============================================================================
class SemanticFilter:
    """Semantic filter for uncertainty-related content"""
    
    def __init__(self, queries: List[str], model_name: str = MODEL_NAME):
        logger.info(f"Initializing Semantic Filter...")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Queries: {len(queries)}")
        
        self.model = SentenceTransformer(model_name)
        self.queries = queries
        self._query_embeddings = None
    
    def _encode_queries(self) -> np.ndarray:
        """Encode queries (cached)"""
        if self._query_embeddings is None:
            logger.info("Encoding queries...")
            self._query_embeddings = self.model.encode(
                self.queries,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        return self._query_embeddings
    
    def compute_scores(self, texts: List[str], batch_size: int = BATCH_SIZE) -> Dict:
        """Compute relevance scores for texts"""
        logger.info(f"Computing scores for {len(texts)} texts...")
        
        query_embeddings = self._encode_queries()
        
        # Encode documents
        logger.info("  Encoding documents...")
        doc_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Compute similarities
        logger.info("  Computing similarities...")
        similarities = cosine_similarity(query_embeddings, doc_embeddings)
        
        # Max and average similarity per document
        max_sim = similarities.max(axis=0)
        avg_sim = similarities.mean(axis=0)
        
        # Convert to percentage scores (0-100)
        scores = (max_sim * 100).clip(0, 100)
        
        logger.info(f"  Score range: {scores.min():.1f} - {scores.max():.1f}")
        logger.info(f"  Mean score: {scores.mean():.1f}")
        
        return {
            'scores': scores,
            'max_similarity': max_sim,
            'avg_similarity': avg_sim
        }

# ============================================================================
# File Reading
# ============================================================================
def read_files(folders: List[Path]) -> pd.DataFrame:
    """Read text files from multiple folders"""
    logger.info(f"Reading files from {len(folders)} folders...")
    
    all_data = []
    
    for folder in folders:
        if not folder.exists():
            logger.debug(f"  Folder not found: {folder.name}")
            continue
        
        txt_files = list(folder.glob("*.txt"))
        pdf_files = list(folder.glob("*.pdf")) if PDF_AVAILABLE else []
        
        logger.info(f"  {folder.name}: {len(txt_files)} TXT, {len(pdf_files)} PDF")
        
        # Read TXT files
        for fpath in txt_files:
            try:
                content = fpath.read_text(encoding='utf-8', errors='ignore')
                
                if len(content.strip()) < 50:
                    continue
                
                all_data.append({
                    'filename': fpath.name,
                    'text': content,
                    'source': folder.name,
                    'file_type': 'txt',
                    'full_path': str(fpath),
                })
            except Exception as e:
                logger.debug(f"    Error reading {fpath.name}: {e}")
        
        # Read PDF files (if no TXT counterpart exists)
        for fpath in pdf_files:
            txt_counterpart = fpath.with_suffix('.txt')
            if txt_counterpart.exists():
                continue  # Skip, already have TXT version
            
            try:
                content = extract_text_from_pdf(fpath)
                
                if len(content.strip()) < 100:
                    continue
                
                all_data.append({
                    'filename': fpath.name,
                    'text': content,
                    'source': folder.name,
                    'file_type': 'pdf',
                    'full_path': str(fpath),
                })
            except Exception as e:
                logger.debug(f"    Error reading PDF {fpath.name}: {e}")
    
    df = pd.DataFrame(all_data)
    
    logger.info(f"\nTotal files loaded: {len(df)}")
    
    if len(df) > 0:
        logger.info("\nBy source:")
        for source, count in df['source'].value_counts().items():
            logger.info(f"  {source}: {count}")
    
    return df

# ============================================================================
# Filtering
# ============================================================================
def filter_texts(df: pd.DataFrame, threshold: float = THRESHOLD) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter texts based on semantic relevance"""
    logger.info(f"\n{'='*60}")
    logger.info("SEMANTIC FILTERING")
    logger.info(f"{'='*60}")
    logger.info(f"Input: {len(df)} texts")
    logger.info(f"Threshold: {threshold}")
    
    # Initialize filter
    sem_filter = SemanticFilter(QUERIES, MODEL_NAME)
    
    # Compute scores
    texts = df['text'].tolist()
    results = sem_filter.compute_scores(texts)
    
    # Add scores to dataframe
    df['semantic_score'] = results['scores']
    df['max_similarity'] = results['max_similarity']
    df['avg_similarity'] = results['avg_similarity']
    
    # Filter
    filtered_df = df[df['semantic_score'] >= threshold].copy()
    
    # Statistics
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Original:  {len(df):>6} texts")
    logger.info(f"Kept:      {len(filtered_df):>6} texts ({len(filtered_df)/len(df)*100:.1f}%)")
    logger.info(f"Removed:   {len(df)-len(filtered_df):>6} texts")
    
    # Per-source statistics
    logger.info("\nBy source:")
    for source in sorted(df['source'].unique()):
        total = len(df[df['source'] == source])
        kept = len(filtered_df[filtered_df['source'] == source])
        pct = kept/total*100 if total > 0 else 0
        logger.info(f"  {source:30s}: {kept:>4}/{total:>4} ({pct:>5.1f}%)")
    
    return filtered_df, df

# ============================================================================
# Save Results
# ============================================================================
def save_filtered_texts(filtered_df: pd.DataFrame, output_folder: Path):
    """Save filtered texts to output folder"""
    import hashlib
    
    logger.info(f"\nSaving filtered texts to {output_folder.name}/")
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    errors = 0
    
    for idx, row in filtered_df.iterrows():
        try:
            # Prepare filename
            filename = row['filename']
            if row['file_type'] == 'pdf':
                filename = Path(filename).stem + '.txt'
            
            # Add source prefix
            source_prefix = row['source'].replace('_', '-')[:15]
            safe_name = f"[{source_prefix}]_{filename}"
            
            # Handle long filenames
            if len(safe_name) > 150:
                name_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
                safe_name = f"[{source_prefix}]_{name_hash}.txt"
            
            output_path = output_folder / safe_name
            
            # Write file
            output_path.write_text(row['text'], encoding='utf-8')
            saved += 1
            
            if saved % 100 == 0:
                logger.info(f"  Progress: {saved}/{len(filtered_df)}")
        
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning(f"  Error saving {idx}: {e}")
    
    logger.info(f"  Saved: {saved} files")
    if errors > 0:
        logger.warning(f"  Errors: {errors}")

def save_statistics(filtered_df: pd.DataFrame, all_df: pd.DataFrame, stats_folder: Path):
    """Save statistics and visualizations"""
    logger.info(f"Saving statistics to {stats_folder.name}/")
    
    stats_folder.mkdir(parents=True, exist_ok=True)
    
    # 1. Save scores as CSV
    all_df[['filename', 'source', 'file_type', 'semantic_score']].to_csv(
        stats_folder / 'all_scores.csv',
        index=False,
        encoding='utf-8-sig'
    )
    
    filtered_df[['filename', 'source', 'file_type', 'semantic_score']].to_csv(
        stats_folder / 'filtered_scores.csv',
        index=False,
        encoding='utf-8-sig'
    )
    
    # 2. Save JSON summary
    summary = {
        'total_input': len(all_df),
        'total_filtered': len(filtered_df),
        'filter_rate_percent': round(len(filtered_df) / len(all_df) * 100, 1) if len(all_df) > 0 else 0,
        'threshold': THRESHOLD,
        'model': MODEL_NAME,
        'score_stats': {
            'mean': round(float(all_df['semantic_score'].mean()), 2),
            'median': round(float(all_df['semantic_score'].median()), 2),
            'min': round(float(all_df['semantic_score'].min()), 2),
            'max': round(float(all_df['semantic_score'].max()), 2),
        },
        'by_source': {}
    }
    
    for source in all_df['source'].unique():
        total = len(all_df[all_df['source'] == source])
        kept = len(filtered_df[filtered_df['source'] == source])
        summary['by_source'][source] = {
            'total': total,
            'kept': kept,
            'rate_percent': round(kept / total * 100, 1) if total > 0 else 0
        }
    
    with open(stats_folder / 'filter_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 3. Create visualization
    create_visualization(all_df, filtered_df, stats_folder)
    
    logger.info("  Statistics saved")

def create_visualization(all_df: pd.DataFrame, filtered_df: pd.DataFrame, stats_folder: Path):
    """Create filtering visualization"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Score distribution
        ax = axes[0, 0]
        ax.hist(all_df['semantic_score'], bins=50, alpha=0.5, label='All', color='gray')
        ax.hist(filtered_df['semantic_score'], bins=50, alpha=0.7, label='Kept', color='green')
        ax.axvline(THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold={THRESHOLD}')
        ax.set_xlabel('Semantic Score')
        ax.set_ylabel('Count')
        ax.set_title('Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. By source (bar chart)
        ax = axes[0, 1]
        sources = sorted(all_df['source'].unique())
        kept = [len(filtered_df[filtered_df['source'] == s]) for s in sources]
        removed = [len(all_df[all_df['source'] == s]) - k for s, k in zip(sources, kept)]
        
        x = np.arange(len(sources))
        ax.bar(x, kept, label='Kept', color='lightgreen')
        ax.bar(x, removed, bottom=kept, label='Removed', color='lightcoral')
        ax.set_xticks(x)
        ax.set_xticklabels([s[:12] for s in sources], rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title('Filtering by Source')
        ax.legend()
        
        # 3. Cumulative distribution
        ax = axes[1, 0]
        sorted_scores = np.sort(all_df['semantic_score'])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
        ax.plot(sorted_scores, cumulative, linewidth=2, color='steelblue')
        ax.axvline(THRESHOLD, color='red', linestyle='--', label=f'Threshold={THRESHOLD}')
        ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Semantic Score')
        ax.set_ylabel('Cumulative %')
        ax.set_title('Cumulative Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Summary stats
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = f"""
        FILTERING SUMMARY
        {'─'*30}
        
        Total Input:     {len(all_df):,}
        Kept:            {len(filtered_df):,} ({len(filtered_df)/len(all_df)*100:.1f}%)
        Removed:         {len(all_df)-len(filtered_df):,}
        
        Threshold:       {THRESHOLD}
        Model:           {MODEL_NAME}
        
        Score Statistics (Kept):
          Mean:          {filtered_df['semantic_score'].mean():.1f}
          Median:        {filtered_df['semantic_score'].median():.1f}
          Min:           {filtered_df['semantic_score'].min():.1f}
          Max:           {filtered_df['semantic_score'].max():.1f}
        """
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(stats_folder / 'filtering_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("  Visualization saved")
        
    except Exception as e:
        logger.warning(f"Visualization error: {e}")

# ============================================================================
# Main
# ============================================================================
def main():
    print(f"\n{'#'*70}")
    print(f"# SEMANTIC FILTER v2.0")
    print(f"# ML Uncertainty Research Pipeline")
    print(f"{'#'*70}\n")
    
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Threshold: {THRESHOLD}")
    logger.info(f"Queries: {len(QUERIES)}")
    
    # Validate input folders
    existing_folders = [f for f in INPUT_FOLDERS if f.exists()]
    
    if not existing_folders:
        logger.error("No input folders found!")
        logger.info("Please run scrapers first to collect data.")
        logger.info(f"Expected folders in: {SCRAPERS_OUTPUT}")
        return
    
    logger.info(f"\nFound {len(existing_folders)}/{len(INPUT_FOLDERS)} input folders")
    
    # Clean previous output
    for folder in [OUTPUT_FOLDER, STATS_FOLDER]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    
    # Read files
    df = read_files(existing_folders)
    
    if len(df) == 0:
        logger.error("No files found in input folders!")
        return
    
    # Filter
    filtered_df, all_df = filter_texts(df, THRESHOLD)
    
    if len(filtered_df) == 0:
        logger.warning("No texts passed the filter!")
        logger.info("Consider lowering the threshold in config.yaml")
        return
    
    # Save results
    save_filtered_texts(filtered_df, OUTPUT_FOLDER)
    save_statistics(filtered_df, all_df, STATS_FOLDER)
    
    # Print top samples
    print(f"\n{'='*60}")
    print("TOP 5 MOST RELEVANT TEXTS")
    print(f"{'='*60}\n")
    
    top5 = filtered_df.nlargest(5, 'semantic_score')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"[{i}] Score: {row['semantic_score']:.1f} | {row['source']} | {row['filename'][:50]}")
        preview = row['text'][:200].replace('\n', ' ')
        print(f"    {preview}...\n")
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"# FILTERING COMPLETE")
    print(f"{'#'*70}")
    print(f"\nOutput:")
    print(f"  Filtered texts: {OUTPUT_FOLDER.relative_to(PIPELINE_DIR)}")
    print(f"  Statistics:     {STATS_FOLDER.relative_to(PIPELINE_DIR)}")
    print(f"\nNext step: Run clustering on filtered data")
    print()


if __name__ == "__main__":
    main()