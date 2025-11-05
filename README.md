# Research Fetcher with Clustering  
Fetch, Analyze, and Cluster Research Data from ArXiv, PubMed, BioRxiv, News, and LinkedIn

## Overview

**Research Fetcher with Clustering** is a Python-based framework for fetching and analyzing academic and professional research data.  
It automatically pulls results from multiple sources, cleans and preprocesses text, and performs advanced clustering to reveal research trends.

This system is designed for teams and researchers who want to **aggregate insights** across fields like AI, bioinformatics, and data science — all in one automated workflow.

---

## ✨ Features

- 🔍 Fetches research from **ArXiv, PubMed, BioRxiv, News, and LinkedIn**
- 🧠 Performs **text preprocessing and clustering** (KMeans, DBSCAN, Hierarchical)
- 📊 Generates **visualizations** and word clouds (optional)
- 📁 Automatically saves **summaries, logs, and cluster reports**
- ⚙️ Supports multi-topic batch runs
- 💬 Fully documented logs with timestamps

---

## 🧩 Hosting Options

| Mode | Description |
|------|--------------|
| 🖥️ **Local Run (Recommended)** | Self-host and execute locally (no API keys required except LinkedIn optional) |
| ☁️ **Cloud (Coming Soon)** | Deploy on server or notebook environment |

---

## ⚙️ System Requirements

### Hardware
- CPU: 4+ cores recommended  
- RAM: 8GB minimum (16GB recommended)  
- Disk: 10GB free space  

### Software
- OS: macOS (10.15+), Ubuntu (20.04+), or Windows 10/11 (via WSL2)
- Python 3.8 or newer  
- Git 2.30+  

### Python Dependencies
The project depends on NLP, clustering, and visualization libraries like:
```
requests, beautifulsoup4, numpy, pandas, scikit-learn,
nltk, matplotlib, seaborn, wordcloud, feedparser
```
You can install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## ⚡ Quick Setup (Recommended)

### For macOS/Linux:
```bash
curl -fsSL https://raw.githubusercontent.com/<yourusername>/research-fetcher-clustering/main/setup.sh -o setup.sh && bash setup.sh
```

### For Windows (PowerShell):
```powershell
iwr https://raw.githubusercontent.com/<yourusername>/research-fetcher-clustering/main/setup.bat -OutFile setup.bat; ./setup.bat
```

Or manually run:
```bash
git clone https://github.com/<yourusername>/research-fetcher-clustering.git
cd research-fetcher-clustering
pip install -r requirements.txt
python main.py
```

---

## 🧠 Example Usage

```bash
python main.py
```

Then enter topics separated by semicolons:
```
machine learning antibody; protein design; drug discovery
```

Optionally, enter your **SerpAPI key** to enable LinkedIn search.

---

## 🧮 Output Overview

Each topic generates:
```
/clustered_research_<timestamp>/
│
├── arxiv_<topic>_COMPLETE.txt
├── pubmed_<topic>_COMPLETE.txt
├── clusters/
│   ├── <source>_<topic>_clustered.csv
│   ├── <source>_<topic>_cluster_report.txt
│   └── visualizations/
└── MASTER_SUMMARY_WITH_CLUSTERS.txt
```

---

## 📊 Visualization Preview

If `matplotlib` and `wordcloud` are installed, the tool produces:
- 2D **PCA cluster scatter plots**
- **Word clouds** for top clusters

---

## 🧱 Project Structure

```
research-fetcher-clustering/
│
├── research_fetcher.py     # Main script
├── requirements.txt        # Dependencies
├── README.md               # Documentation
└── /output/                # Generated results
```

---

## 💡 Notes
- To include LinkedIn scraping, set your **SerpAPI key** in the prompt or save it in a `.env` file.
- Cluster sizes can be auto-detected or manually specified.
- Use this project for academic and educational purposes only.

---

