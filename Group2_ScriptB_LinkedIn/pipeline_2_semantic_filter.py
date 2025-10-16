import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
# import seaborn as sns
from typing import List, Dict
import json

# ============================================================================
# 配置
# ============================================================================
INPUT_FOLDERS = [
    "linkedin_posts",
    "Reddit_posts"
]
OUTPUT_FOLDER = "filtered_posts"
STATS_FOLDER = "filter_stats"

# 语义过滤参数
MODEL_NAME = 'all-MiniLM-L6-v2'
THRESHOLD = 30.0  # 可调整: 20(宽松) - 30(平衡) - 40(严格)
BATCH_SIZE = 32

# 不确定性主题查询
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
# 清空旧结果 ⭐ 新增
# ============================================================================
def clean_previous_results():
    """清空上次运行的结果"""
    folders_to_clean = [OUTPUT_FOLDER, STATS_FOLDER]
    
    print(f"\n🧹 Cleaning previous results...")
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"   ✓ Removed old folder: {folder}/")
            except Exception as e:
                print(f"   ⚠️  Could not remove {folder}/: {e}")
        
        # 重新创建空文件夹
        os.makedirs(folder, exist_ok=True)
        print(f"   ✓ Created clean folder: {folder}/")
    print()

# ============================================================================
# 语义过滤器
# ============================================================================
class UncertaintySemanticFilter:
    """不确定性主题的语义过滤器"""
    
    def __init__(self, queries: List[str], model_name: str = MODEL_NAME):
        print(f"🔧 Initializing Semantic Filter...")
        print(f"   Model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.queries = queries
        self._query_embeddings = None
        
    def _encode_queries(self):
        """编码查询（只执行一次）"""
        if self._query_embeddings is None:
            print(f"📝 Encoding {len(self.queries)} uncertainty queries...")
            self._query_embeddings = self.model.encode(
                self.queries,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            print("   ✓ Queries encoded")
        return self._query_embeddings
    
    def compute_scores(self, texts: List[str], batch_size: int = BATCH_SIZE):
        """计算文本相关性分数"""
        print(f"\n🔍 Computing semantic scores for {len(texts)} texts...")
        
        # 编码查询
        query_embeddings = self._encode_queries()
        
        # 编码文档
        print("   Encoding documents...")
        doc_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # 计算相似度
        print("   Computing similarities...")
        similarities = cosine_similarity(query_embeddings, doc_embeddings)
        
        # 使用最大相似度作为分数
        max_similarity = similarities.max(axis=0)
        avg_similarity = similarities.mean(axis=0)
        
        # 转换为 0-100 分数
        scores = (max_similarity * 100).clip(0, 100)
        
        print(f"   ✓ Score range: {scores.min():.2f} - {scores.max():.2f}")
        print(f"   ✓ Mean score: {scores.mean():.2f}")
        
        return {
            'scores': scores,
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity
        }

# ============================================================================
# 文本读取
# ============================================================================
def read_text_files(folders: List[str]) -> pd.DataFrame:
    """
    从多个文件夹读取文本文件
    
    返回: DataFrame with columns [filename, text, source]
    """
    print(f"\n📂 Reading text files from {len(folders)} folders...")
    
    all_data = []
    for folder in folders:
        if not os.path.exists(folder):
            print(f"   ⚠️  Folder not found: {folder}, skipping")
            continue
        
        folder_path = Path(folder)
        txt_files = list(folder_path.glob("*.txt"))
        print(f"   📁 {folder}: {len(txt_files)} files")
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_data.append({
                        'filename': file_path.name,
                        'text': content,
                        'source': folder
                    })
            except Exception as e:
                print(f"      ❌ Error reading {file_path.name}: {e}")
    
    df = pd.DataFrame(all_data)
    print(f"\n✅ Total texts loaded: {len(df)}")
    if len(df) > 0:
        print(f"   Sources: {df['source'].value_counts().to_dict()}")
    
    return df

# ============================================================================
# 过滤函数
# ============================================================================
def filter_texts(df: pd.DataFrame, threshold: float = THRESHOLD):
    """
    基于语义分数过滤文本
    """
    print(f"\n{'='*70}")
    print(f"SEMANTIC FILTERING")
    print(f"{'='*70}")
    print(f"Input texts: {len(df)}")
    print(f"Threshold: {threshold}")
    print(f"{'='*70}\n")
    
    # 初始化过滤器
    filter_model = UncertaintySemanticFilter(UNCERTAINTY_QUERIES)
    
    # 计算分数
    texts = df['text'].tolist()
    score_results = filter_model.compute_scores(texts)
    
    # 添加分数到 DataFrame
    df['semantic_score'] = score_results['scores']
    df['max_similarity'] = score_results['max_similarity']
    df['avg_similarity'] = score_results['avg_similarity']
    
    # 应用阈值
    filtered_df = df[df['semantic_score'] >= threshold].copy()
    
    # 统计
    print(f"\n{'='*70}")
    print(f"FILTERING RESULTS")
    print(f"{'='*70}")
    print(f"Original:  {len(df):>6} texts")
    print(f"Kept:      {len(filtered_df):>6} texts ({len(filtered_df)/len(df)*100:>5.1f}%)")
    print(f"Removed:   {len(df)-len(filtered_df):>6} texts ({(len(df)-len(filtered_df))/len(df)*100:>5.1f}%)")
    print(f"{'='*70}\n")
    
    # 按来源统计
    if len(df) > 0:
        print("Filtering by source:")
        for source in df['source'].unique():
            source_total = len(df[df['source'] == source])
            source_kept = len(filtered_df[filtered_df['source'] == source])
            print(f"  {source}: {source_kept}/{source_total} ({source_kept/source_total*100:.1f}%)")
    
    return filtered_df, df

# ============================================================================
# 保存结果
# ============================================================================
def save_filtered_texts(filtered_df: pd.DataFrame, output_folder: str):
    """保存过滤后的文本"""
    print(f"\n💾 Saving filtered texts to {output_folder}/")
    
    # 保存每个文本文件
    for idx, row in filtered_df.iterrows():
        output_path = os.path.join(output_folder, row['filename'])
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(row['text'])
    
    print(f"   ✓ Saved {len(filtered_df)} files")

def save_statistics(filtered_df: pd.DataFrame, all_df: pd.DataFrame, stats_folder: str):
    """保存统计信息和可视化"""
    print(f"\n📊 Generating statistics and visualizations...")
    
    # 1. 保存分数 CSV
    all_df[['filename', 'source', 'semantic_score', 'max_similarity', 'avg_similarity']].to_csv(
        os.path.join(stats_folder, 'all_scores.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    
    filtered_df[['filename', 'source', 'semantic_score', 'max_similarity', 'avg_similarity']].to_csv(
        os.path.join(stats_folder, 'filtered_scores.csv'),
        index=False,
        encoding='utf-8-sig'
    )
    
    # 2. 保存 JSON 摘要
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
    
    # 3. 生成可视化
    generate_visualizations(all_df, filtered_df, stats_folder)
    
    print(f"   ✓ Statistics saved to {stats_folder}/")

# ============================================================================
# 可视化（不使用 seaborn）⭐ 修改
# ============================================================================
def generate_visualizations(all_df: pd.DataFrame, filtered_df: pd.DataFrame, stats_folder: str):
    """生成可视化图表（纯 matplotlib）"""
    
    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.facecolor'] = 'white'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 分数分布直方图
    ax = axes[0, 0]
    ax.hist(all_df['semantic_score'], bins=50, alpha=0.6, color='gray', label='All', edgecolor='black')
    ax.hist(filtered_df['semantic_score'], bins=50, alpha=0.8, color='green', label='Kept', edgecolor='black')
    ax.axvline(THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold={THRESHOLD}')
    ax.set_xlabel('Semantic Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 累积分布
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
    
    # 3. 按来源统计
    ax = axes[0, 2]
    source_stats = []
    for source in all_df['source'].unique():
        total = len(all_df[all_df['source'] == source])
        kept = len(filtered_df[filtered_df['source'] == source])
        source_stats.append({'Source': source, 'Kept': kept, 'Removed': total - kept})
    
    if len(source_stats) > 0:
        stats_df = pd.DataFrame(source_stats)
        x = np.arange(len(stats_df))
        width = 0.6
        
        ax.bar(x, stats_df['Kept'], width, label='Kept', color='lightgreen')
        ax.bar(x, stats_df['Removed'], width, bottom=stats_df['Kept'], label='Removed', color='lightcoral')
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Filtering by Source', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df['Source'], rotation=45, ha='right')
        ax.legend()
    
    # 4. 箱线图对比
    ax = axes[1, 0]
    kept_scores = filtered_df['semantic_score'].values
    removed_scores = all_df[all_df['semantic_score'] < THRESHOLD]['semantic_score'].values
    
    bp = ax.boxplot([kept_scores, removed_scores], labels=['Kept', 'Removed'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Semantic Score', fontsize=12)
    ax.set_title('Score Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. 分数 vs 来源（分组条形图）
    ax = axes[1, 1]
    sources = all_df['source'].unique()
    kept_means = []
    removed_means = []
    
    for source in sources:
        source_df = all_df[all_df['source'] == source]
        kept_mask = source_df['semantic_score'] >= THRESHOLD
        
        kept_mean = source_df[kept_mask]['semantic_score'].mean() if kept_mask.sum() > 0 else 0
        removed_mean = source_df[~kept_mask]['semantic_score'].mean() if (~kept_mask).sum() > 0 else 0
        
        kept_means.append(kept_mean)
        removed_means.append(removed_mean)
    
    x = np.arange(len(sources))
    width = 0.35
    
    ax.bar(x - width/2, kept_means, width, label='Kept', color='lightgreen')
    ax.bar(x + width/2, removed_means, width, label='Removed', color='lightcoral')
    
    ax.set_ylabel('Mean Semantic Score', fontsize=12)
    ax.set_xlabel('Source', fontsize=12)
    ax.set_title('Average Score by Source & Status', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. 统计表格
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
    
    print(f"   ✓ Visualization saved: filtering_analysis.png")

# ============================================================================
# 显示样本
# ============================================================================
def print_sample_texts(filtered_df: pd.DataFrame, n_samples: int = 5):
    """打印高分样本"""
    if len(filtered_df) == 0:
        print("\n⚠️  No texts passed the filter!")
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
# 主函数
# ============================================================================
def main():
    print(f"\n{'#'*70}")
    print(f"# PIPELINE 2: SEMANTIC FILTERING FOR UNCERTAINTY TOPICS")
    print(f"{'#'*70}\n")
    
    # Step 0: 清空旧结果 ⭐ 新增
    clean_previous_results()
    
    # Step 1: 读取文本
    df = read_text_files(INPUT_FOLDERS)
    
    if len(df) == 0:
        print("❌ No texts found! Please check input folders.")
        print(f"\n   Expected folders:")
        for folder in INPUT_FOLDERS:
            print(f"   - {folder}/")
        return
    
    # Step 2: 语义过滤
    filtered_df, all_df = filter_texts(df, threshold=THRESHOLD)
    
    # Step 3: 保存过滤后的文本
    save_filtered_texts(filtered_df, OUTPUT_FOLDER)
    
    # Step 4: 保存统计和可视化
    save_statistics(filtered_df, all_df, STATS_FOLDER)
    
    # Step 5: 显示样本
    print_sample_texts(filtered_df, n_samples=5)
    
    # 总结
    print(f"\n{'#'*70}")
    print(f"# FILTERING COMPLETED SUCCESSFULLY")
    print(f"{'#'*70}")
    print(f"\n📁 Output folders:")
    print(f"   - Filtered texts: {OUTPUT_FOLDER}/")
    print(f"   - Statistics:     {STATS_FOLDER}/")
    print(f"\n🎯 Next step: Run pipeline_3_cluster.py for clustering\n")

if __name__ == "__main__":
    main()
