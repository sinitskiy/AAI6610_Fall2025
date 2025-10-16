# 不确定性估计主题聚类项目
# Uncertainty Estimation Topic Clustering Project

LinkedIn 和 Reddit 收集关于"不确定性估计"的帖子，然后用聚类算法分析出不同的研究主题。
LinkedIn and Reddit collect posts related to "uncertainty estimation", and then use clustering algorithms to analyze different research topics.

## 0.文件结构
## 0.File Structure
Group_2_Script_B
│
├── requirements.txt                           # Python依赖包列表 | Python Dependencies List
├── linkedin_cookies.json                      # LinkedIn登录凭证 | LinkedIn Login Credentials
├── READ.md                                    # 项目说明文档 | Project Documentation
│
├── pipeline_0_linkedin_cookie.py              # Step 0: 获取LinkedIn Cookies | Get LinkedIn Cookies
├── pipeline_1_crawler_linkedin.py             # Step 1a: LinkedIn帖子采集 | LinkedIn Post Scraper
├── pipeline_1_crawler_Reddit.py               # Step 1b: Reddit帖子采集 | Reddit Post Scraper
├── pipeline_2_semantic_filter.py              # Step 2: 语义过滤器 | Semantic Filter
├── pipeline_3_cluster.py                      # Step 3a: 聚类分析 | Clustering Analysis
├── pipeline_3_cluster_print.py                # Step 3b: 聚类可视化 | Clustering Visualization
│
├── linkedin_posts/                            # LinkedIn原始帖子 | Raw LinkedIn Posts
│   ├── a1b2c3d4e5f6.txt                      # 帖子1 | Post 1
│   ├── f6e5d4c3b2a1.txt                      # 帖子2 | Post 2
│   └── ... (500+ files)                      # 数百个帖子文件 | Hundreds of post files
│
├── Reddit_posts/                              # Reddit/StackExchange原始帖子 | Raw Reddit/StackExchange Posts
│   ├── uncertainty_in_machine_learning.txt   # Reddit帖子示例 | Reddit post example
│   ├── model_uncertainty_quantification.txt  # StackExchange问答示例 | StackExchange Q&A example
│   └── ... (200-500 files)                   # 数百个帖子文件 | Hundreds of post files
│
├── filter_stats/                              # 过滤统计数据 | Filtering Statistics
│   ├── filtering_analysis.png                # 过滤效果可视化图 | Filtering Effect Visualization
│   ├── all_scores.csv                        # 所有帖子评分（过滤前）| All Posts Scores (Before Filter)
│   ├── filtered_scores.csv                   # 通过过滤的帖子评分 | Filtered Posts Scores
│   └── filter_summary.json                   # 过滤统计摘要 | Filter Summary Statistics
│
├── filtered_posts/                            # 过滤后的高质量帖子 | Filtered High-Quality Posts
│   ├── a1b2c3d4e5f6.txt                      # 高质量帖子1 | High-quality post 1
│   ├── i9j0k1l2m3n4.txt                      # 高质量帖子2 | High-quality post 2
│   └── ... (100-300 files)                   # 过滤后保留的帖子 | Posts retained after filtering
│
├── cluster_output/                            # 聚类分析结果 | Clustering Analysis Results
│   ├── embeddings.pkl                        # 文本向量数据（3072维）| Text Embeddings (3072-dim)
│   │
│   ├── hdbscan_umap2d.png                    # HDBSCAN可视化图 | HDBSCAN Visualization
│   ├── HDBSCAN_clusters.csv                  # HDBSCAN聚类映射表 | HDBSCAN Cluster Mapping
│   ├── HDBSCAN_representatives.txt           # HDBSCAN簇摘要（关键词+代表文本）| HDBSCAN Cluster Summary
│   ├── HDBSCAN_meta.json                     # HDBSCAN完整元数据 | HDBSCAN Full Metadata
│   │
│   ├── kmeans_k2_umap2d.png                  # KMeans可视化图（K=2）| KMeans Visualization (K=2)
│   ├── KMeans_K2_clusters.csv                # KMeans聚类映射表 | KMeans Cluster Mapping
│   ├── KMeans_K2_representatives.txt         # KMeans簇摘要 | KMeans Cluster Summary
│   └── KMeans_K2_meta.json                   # KMeans完整元数据 | KMeans Full Metadata
│
└── cluster_visualizations/                    # 聚类多维度可视化 | Multi-dimensional Cluster Visualizations
    ├── HDBSCAN_pca2d.png                     # HDBSCAN-PCA降维图 | HDBSCAN-PCA Dimensionality Reduction
    ├── HDBSCAN_tsne.png                      # HDBSCAN-tSNE降维图 | HDBSCAN-tSNE Dimensionality Reduction
    ├── HDBSCAN_umap.png                      # HDBSCAN-UMAP降维图 | HDBSCAN-UMAP Dimensionality Reduction
    ├── KMeans_K2_pca2d.png                   # KMeans-PCA降维图 | KMeans-PCA Dimensionality Reduction
    ├── KMeans_K2_tsne.png                    # KMeans-tSNE降维图 | KMeans-tSNE Dimensionality Reduction
    └── KMeans_K2_umap.png                    # KMeans-UMAP降维图 | KMeans-UMAP Dimensionality Reduction


### 1. 安装依赖 Install dependencies
```bash
pip install -r requirements.txt
```

### 2. 运行流程 Operation process
```bash
# 第 1 步：爬数据
# Step 1: Collect Data.
# The data is stored in "linkedin_posts".
python pipeline_1_crawler_linkedin.py
# The data is stored in "Reddit_posts".
python pipeline_1_crawler_Reddit.py

# 第 2 步：过滤数据
# Step 2: Filter the data
#The data is input from the "linkedin_posts" and "Reddit_posts" sources and is then output to the "filtered_posts" destination.
#The "filter_stats" folder contains the filtering statistics charts.
python pipeline_2_semantic_filter.py

# 第 3 步：聚类
# Step 3: Clustering
#Input is "filtered_posts", and output is "cluster_output".
python pipeline_3_cluster.py

# 第 4 步：生成图表
# Step 4: Generate Charts
# The generated graph is in the "cluster_visualizations" folder.
python pipeline_4_visualizations.py
```

### Pipeline 1: 数据采集 🕷️
### Pipeline 1: Data Collection
**Function**: Scrape posts related to "uncertainty estimation" from LinkedIn and Reddit
- Need: Google Custom Search API,SEARCH_ENGINE_ID,linkedin account.
- Use the script "preliminary_pipeline_0_linkedin_cookie.py" to obtain the LinkedIn account. JSON file
# Enter the API key here:
```bash
GOOGLE_API_KEY = "-"   # Google Custom Search API Key
SEARCH_ENGINE_ID = "-" # Google Custom Search Engine ID
```
- `pipeline_1_crawler_linkedin.py`
Need: Reddit API credentials.
- `pipeline_1_crawler_Reddit.py`
**output**：
- `linkedin_posts/` - 451 .txt
- `Reddit_posts/` - 1068 .txt

### Pipeline 2: 语义过滤
### Pipeline 2: Semantic Filtering
**Function**:
From 1519 posts, filter out the truly relevant ones (eliminating chats, advertisements, etc.)
- Use an AI model to calculate the similarity between each post and the "uncertainty estimation" topic
- Only retain posts with a similarity score greater than 30 points
**output**：
- `filtered_posts/` - 532 high-quality posts
- `filter_stats/` - Filter statistics charts
**setting**：
```python
# If you want stricter filtering, change this number.
# The default is 30. Changing it to 40 will make the filtering even stricter.
THRESHOLD = 30.0  
```

### Pipeline 3: 聚类分析 
### Pipeline 3: Cluster Analysis
**Function**:
Automatically categorize 532 posts into different topics
- OpenAI API: Convert text into numbers (embedding)
- UMAP: Dimensionality reduction (reduce from 3072 dimensions to 50 dimensions)
- HDBSCAN: Automatically identify 16 topics
- Need:OpenAI API
# Enter the API key here:
```bash
export OPENAI_API_KEY="你的密钥"
```
**output**
- `cluster_output/HDBSCAN_clusters.csv` - Which topic each post belongs to
- `cluster_output/HDBSCAN_representatives.txt` - Keywords for each topic and representative posts
- `cluster_output/hdbscan_umap2d.png` - Visualization of clustering results

### Pipeline 4: 可视化 
### Pipeline 4: Visualization
cluster_visualizations/                    # 聚类多维度可视化 | Multi-dimensional Cluster Visualizations
    ├── HDBSCAN_pca2d.png                     # HDBSCAN-PCA降维图 | HDBSCAN-PCA Dimensionality Reduction
    ├── HDBSCAN_tsne.png                      # HDBSCAN-tSNE降维图 | HDBSCAN-tSNE Dimensionality Reduction
    ├── HDBSCAN_umap.png                      # HDBSCAN-UMAP降维图 | HDBSCAN-UMAP Dimensionality Reduction
    ├── KMeans_K2_pca2d.png                   # KMeans-PCA降维图 | KMeans-PCA Dimensionality Reduction
    ├── KMeans_K2_tsne.png                    # KMeans-tSNE降维图 | KMeans-tSNE Dimensionality Reduction
    └── KMeans_K2_umap.png                    # KMeans-UMAP降维图 | KMeans-UMAP Dimensionality Reduction



