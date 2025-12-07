# AAI6610 Fall 2025 - Pipeline 2

**Multi-Source Research Aggregation Pipeline with ML-Powered Clustering**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**Pipeline 2** is a comprehensive research aggregation pipeline that collects academic papers and related content from multiple sources, then uses machine learning to automatically cluster and organize them by semantic similarity. Built for the AAI6610 Applied Machine Learning course at Northeastern University.

---

## What This Pipeline Does

Pipeline 2 performs the following automated workflow:

1. **Data Collection**: Fetches research papers from multiple academic and professional sources
   - ArXiv (preprints in CS, Biology, Statistics)
   - PubMed (biomedical literature)
   - BioRxiv (biology preprints)
   - OpenAlex (peer-reviewed papers)
   - News aggregators (Google News, Bing News)
   - LinkedIn (professional discussions)
   - Reddit (community discussions)

2. **Text Processing**: Standardizes and preprocesses collected data
   - Extracts text from PDFs
   - Normalizes metadata across sources
   - Deduplicates papers appearing in multiple sources

3. **Semantic Filtering**: Uses sentence transformers to filter relevant content
   - Employs `all-MiniLM-L6-v2` for semantic similarity
   - Filters based on configurable relevance thresholds

4. **ML Clustering**: Groups similar papers using unsupervised learning
   - K-Means clustering with automatic k selection
   - DBSCAN for density-based clustering
   - Hierarchical (Agglomerative) clustering
   - HDBSCAN for variable density clusters

5. **Visualization & Reporting**: Generates comprehensive outputs
   - 2D PCA/UMAP cluster visualizations
   - Word clouds for topic identification
   - Detailed cluster reports and summaries

---

## Project Structure

```
AAI6610_Fall2025/
├── pipeline2/                    # All pipeline code
│   ├── app.py                    # Flask web server
│   ├── multisearchfinal.py       # Main pipeline orchestrator
│   ├── config.yaml               # Configuration settings
│   ├── requirements.txt          # Python dependencies
│   ├── .env.example              # Environment variables template
│   ├── index.html                # Web interface
│   │
│   ├── scrapers/                 # Data collection modules
│   │   └── codes/
│   │       ├── scraper_arxiv.py
│   │       ├── scraper_biorxiv.py
│   │       ├── scraper_openalex.py
│   │       ├── scraper_news.py
│   │       ├── scraper_linkedin.py
│   │       ├── scraper_reddit.py
│   │       └── pdf_to_txt.py
│   │
│   ├── clustering/               # ML clustering modules
│   │   └── codes/
│   │       ├── cluster_engine.py
│   │       ├── semantic_filter.py
│   │       └── visualizer.py
│   │
│   └── main_control/             # Configuration management
│       ├── config_loader.py
│       ├── gui.py
│       └── update_api_keys.py
│
├── .gitignore
├── LICENSE
└── README.md                     # This file
```

---

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/sinitskiy/AAI6610_Fall2025.git
cd AAI6610_Fall2025
git checkout whole_pipeline2
```

### Step 2: Create Virtual Environment (Recommended)

```bash
cd pipeline2
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your API keys (optional - pipeline works without them)
nano .env  # or use any text editor
```

### Step 5: Run the Pipeline

**Option A: Web Interface (Recommended)**
```bash
python app.py
# Open http://localhost:5000 in your browser
```

**Option B: Command Line**
```bash
python multisearchfinal.py
```

---

## Dependencies

### Core Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| requests | ≥2.31.0 | HTTP requests for API calls |
| beautifulsoup4 | ≥4.12.2 | HTML/XML parsing |
| lxml | ≥4.9.3 | XML processing |
| numpy | ≥1.24.0 | Numerical computations |
| pandas | ≥2.0.0 | Data manipulation |
| scikit-learn | ≥1.3.0 | ML clustering algorithms |
| nltk | ≥3.8.1 | Natural language processing |
| flask | ≥3.0.0 | Web server |
| flask-cors | ≥4.0.0 | Cross-origin requests |

### Visualization (Optional)

| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | ≥3.7.1 | Plotting |
| seaborn | ≥0.13.0 | Statistical visualization |
| wordcloud | ≥1.9.2 | Word cloud generation |
| pillow | ≥10.0.0 | Image processing |

### Additional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scipy | ≥1.10.0 | Scientific computing |
| feedparser | ≥6.0.10 | RSS feed parsing |

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Expected Inputs

### Configuration File (`config.yaml`)

The pipeline is configured via `config.yaml`:

```yaml
topic: "your research topic"

scrapers:
  arxiv:
    enabled: true
    output_folder: arxiv_papers
    timeout: 1800
  openalex:
    enabled: true
    output_folder: openalex_papers
    timeout: 2400
  # ... other sources

clustering:
  min_k: 5
  max_k: 20
  top_keywords: 10
```

### Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| OPENAI_API_KEY | Optional | For advanced embeddings |
| GOOGLE_API_KEY | Optional | For LinkedIn/News search |
| GOOGLE_SEARCH_ENGINE_ID | Optional | For custom search |
| REDDIT_CLIENT_ID | Optional | For Reddit scraping |
| REDDIT_CLIENT_SECRET | Optional | For Reddit scraping |

### Web Interface Inputs

- **Topics**: Research keywords (e.g., "machine learning antibody")
- **Sources**: Toggle individual data sources on/off
- **Limits**: Number of items per source (default: 100 for quick runs)
- **Clustering Algorithm**: K-Means, DBSCAN, or Hierarchical

---

## Expected Outputs

### Output Directory Structure

```
final_output/research_output_YYYYMMDD_HHMMSS/
│
├── ArXiv_<topic>_COMPLETE.txt           # Raw ArXiv papers
├── PubMed_<topic>_COMPLETE.txt          # Raw PubMed papers
├── BioRxiv_<topic>_COMPLETE.txt         # Raw BioRxiv papers
├── OpenAlex_<topic>_COMPLETE.txt        # Raw OpenAlex papers
├── News_<topic>_COMPLETE.txt            # News articles
├── LinkedIn_<topic>_COMPLETE.txt        # LinkedIn posts
│
├── clusters/
│   ├── <source>_<topic>_clustered.csv   # Papers with cluster labels
│   ├── <source>_<topic>_cluster_summary.json
│   ├── <source>_<topic>_cluster_report.txt
│   └── visualizations/
│       ├── <source>_clusters_pca.png    # PCA visualization
│       └── <source>_wordcloud.png       # Topic word clouds
│
├── logs/
│   └── main_log_YYYYMMDD_HHMMSS.txt     # Execution logs
│
└── MASTER_SUMMARY_WITH_CLUSTERS.txt     # Overall summary
```

### Output File Formats

| File Type | Format | Description |
|-----------|--------|-------------|
| `*_COMPLETE.txt` | Plain text | Raw collected papers with metadata |
| `*_clustered.csv` | CSV | Papers with cluster assignments (Excel-compatible) |
| `*_cluster_report.txt` | Plain text | Human-readable cluster analysis |
| `*_cluster_summary.json` | JSON | Structured cluster metadata |
| `*.png` | Image | Visualizations |

---

## Runtime Configuration

### Default Settings (Quick Run: ~10-15 minutes)

The default configuration limits data collection for faster execution:

- Date range: November-December 2025
- Max items per source: 100
- All scrapers: disabled by default (enable as needed)

### Full Run Settings

For comprehensive data collection, modify `config.yaml`:

```yaml
scrapers:
  arxiv:
    enabled: true
  openalex:
    enabled: true
  biorxiv:
    enabled: true
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: flask` | Run `pip install flask flask-cors` |
| Port 5000 in use | Disable AirPlay Receiver (macOS) or change port in `app.py` |
| API rate limits | Increase delay in `config.yaml` or reduce item limits |
| Empty results | Check date range and search query in config |
| Import errors | Ensure virtual environment is activated |

---

## Authors & Acknowledgments

**Course**: AAI6610 - Applied Machine Learning  
**Instructor**: Prof. Anton Sinitskiy  
**University**: Northeastern University  
**Term**: Fall 2025

### Pipeline Contributors

- **Pipeline 2 (Pipeline 2)**: Ruthvik Bandari
  - OpenReview/OpenAlex integration
  - Peer-reviewed paper scraping
  - Data standardization and deduplication

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Pipeline Name

**Pipeline 2** - Multi-source research aggregation pipeline of aggregating and analyzing peer-reviewed research papers from multiple scholarly sources.

Alternative suggestions:
- **Nexus** - Connection point for multiple research sources
- **Curator** - Collecting and organizing academic knowledge
- **Beacon** - Illuminating research trends

---

## Quick Test

```bash
cd pipeline2
source venv/bin/activate
python3 -c "from multisearchfinal import MultiTopicResearchFetcherWithClustering; print('✅ Import successful!')"
python app.py
# Open http://localhost:5000
# Use default topic with ArXiv only, limit 100 items
# Should complete in ~2-5 minutes
```
