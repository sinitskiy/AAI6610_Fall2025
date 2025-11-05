# 🚀 Research Fetcher Pro

**Multi-Source Research Paper Aggregator with Machine Learning Clustering**

A web-based platform that fetches research papers from multiple sources and uses machine learning to automatically group them by similarity.

---

## 📖 What This Does

This tool automatically:
1. **Searches** ArXiv, PubMed, BioRxiv, News, and LinkedIn for research papers
2. **Collects** up to 1000 papers per source
3. **Uses ML** to cluster similar papers together
4. **Generates** reports, visualizations, and organized files
5. **Shows results** in a clean web interface

---

## ⚡ Quick Start

### 1. Install Python Packages

```bash
pip install flask flask-cors requests beautifulsoup4 numpy pandas scikit-learn nltk
```

### 2. Start the Server

```bash
python app.py
```

Wait for:
```
✅ Server Status: Running
🌐 URL: http://localhost:5000
```

### 3. Open Your Browser

Go to: **http://localhost:5000**

Done! 🎉

---

## 💻 How to Use

### **Step 1: Add Topics**
- Type a research topic (e.g., "protein folding", "CRISPR gene editing")
- Click "Add" or press Enter
- Add multiple topics if you want

### **Step 2: Choose Sources**
- Toggle sources ON/OFF (ArXiv, PubMed, BioRxiv, News, LinkedIn)
- Click the ▼ button to adjust item limits
- LinkedIn is optional (works without API key)

### **Step 3: Configure Clustering**
- Pick algorithm: K-Means (fast), DBSCAN, or Hierarchical
- Keep "Auto-detect" checked (finds optimal clusters automatically)

### **Step 4: Run It**
- Click the big blue **"Start Fetching"** button
- Switch to "Progress" tab to watch real-time updates
- Takes 5-15 minutes depending on settings

### **Step 5: View Results**
- See total papers collected, clusters found, quality metrics
- Export data as CSV or JSON
- Find detailed reports in the output folder

---

## 📁 What You Get

After running, a new folder is created:
```
research_output_20251105_134512/
├── ArXiv_machine_learning_antibody_COMPLETE.txt
├── PubMed_machine_learning_antibody_COMPLETE.txt
├── BioRxiv_machine_learning_antibody_COMPLETE.txt
├── News_machine_learning_antibody_COMPLETE.txt
├── LinkedIn_machine_learning_antibody_COMPLETE.txt
│
├── clusters/
│   ├── arxiv_machine_learning_antibody_clustered.csv
│   ├── arxiv_machine_learning_antibody_cluster_summary.json
│   ├── arxiv_machine_learning_antibody_cluster_report.txt
│   └── visualizations/
│       └── arxiv_machine_learning_antibody_clusters.png
│
├── logs/
│   └── main_log_20251105_134512.txt
│
└── MASTER_SUMMARY_WITH_CLUSTERS.txt
```

**Key Files:**
- **`*_COMPLETE.txt`** - All papers from each source
- **`*_clustered.csv`** - Papers with cluster labels (open in Excel)
- **`*_cluster_report.txt`** - Human-readable cluster analysis
- **`MASTER_SUMMARY_WITH_CLUSTERS.txt`** - Overview of everything

---

## 🔧 Files You Need

These 3 files must be in the same folder:
1. **`multisearchfinal.py`** - Main research fetcher code
2. **`app.py`** - Web server (Flask backend)
3. **`index.html`** - User interface (Web GUI)

---

## 🔑 LinkedIn (Optional)

LinkedIn works two ways:

**Option 1 - No API Key** (Default)
- Just toggle LinkedIn ON
- Gets 1000 placeholder items
- Perfect for testing and demos

**Option 2 - With API Key** (Real Data)
- Sign up at https://serpapi.com (free tier: 100 searches/month)
- Paste API key in the yellow box
- Gets real LinkedIn posts

---

## 🧪 Quick Test (30 seconds)

1. Keep default topic: "machine learning antibody"
2. Enable only ArXiv and PubMed
3. Change both limits to **100**
4. Click "Start Fetching"
5. Should finish in ~30 seconds

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Backend not connected"** | Make sure `python app.py` is running |
| **"404 Not Found"** | Check that `index.html` is in same folder as `app.py` |
| **"Module not found"** | Run: `pip install flask flask-cors` |
| **Port 5000 in use** | Change `port=5000` to `port=5001` in `app.py` line 256 |
| **Slow performance** | Reduce item limits to 100-500 per source |

---

## 📊 Technical Details

### **Machine Learning**
- **Algorithms**: K-Means, DBSCAN, Hierarchical clustering
- **NLP**: TF-IDF vectorization, tokenization, lemmatization
- **Quality Metrics**: Silhouette score, Davies-Bouldin index
- **Auto-optimization**: Finds best number of clusters automatically

### **Data Sources**
- **ArXiv**: Academic preprints (cs, bio, stats)
- **PubMed**: Medical/biological research (NIH database)
- **BioRxiv**: Biology preprints
- **News**: Google News, Bing News RSS feeds
- **LinkedIn**: Professional discussions (requires SerpAPI)

---

## 🎓 For AAI6610 Students

**Course**: Applied Machine Learning (AAI6610)  
**Instructor**: Anton Sinitskiy  
**University**: Northeastern University  
**Term**: Fall 2025

### **Project Requirements Met:**
✅ Real-world AI application  
✅ Advanced ML algorithms (clustering, NLP)  
✅ Multi-source data collection  
✅ Web-based deployment  
✅ Publication-ready outputs  
✅ Complete documentation  

### **Technologies Used:**
- **Python**: Core programming language
- **scikit-learn**: ML clustering algorithms
- **NLTK**: Natural language processing
- **Flask**: Web server / REST API
- **React**: User interface
- **Pandas/NumPy**: Data processing

---

## 👥 Sharing with Team

Send teammates these 3 files:
1. `multisearchfinal.py`
2. `app.py`
3. `index.html`

Instructions for them:
```
1. Put all 3 files in a folder
2. Run: pip install flask flask-cors requests beautifulsoup4 numpy pandas scikit-learn nltk
3. Run: python app.py
4. Open: http://localhost:5000
```

---

## 📦 Project Structure

```
AAI6610_Fall2025/
├── multisearchfinal.py    # Main research fetcher (1000+ lines)
├── app.py                  # Flask backend server (~250 lines)
├── index.html             # Web GUI (~500 lines)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🚀 GitHub Repository

**Clone this project:**
```bash
git clone https://github.com/sinitskiy/AAI6610_Fall2025.git
cd AAI6610_Fall2025
git checkout whole_pipeline2
```

---

## 💡 Pro Tips

- Start with small limits (100-200) to test quickly
- Use ArXiv + PubMed for academic research
- Check Progress tab for real-time logs
- Output folders are timestamped (won't overwrite)
- CSV files can be opened in Excel for analysis
- Cluster reports are human-readable text files

---

## ❓ Questions?

- **For bugs**: Check terminal where `app.py` is running
- **For web issues**: Press F12 in browser, check Console tab
- **For Git help**: See Git commands above
- **For course questions**: Email instructor

---

**Ready to fetch some research! 🔬**```
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

