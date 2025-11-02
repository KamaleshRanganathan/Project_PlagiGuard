# PlagiGuard — AI Plagiarism Detector  
**Catch copies. Protect originality.**

An **AI-powered plagiarism detection tool** that compares student submissions using **four advanced NLP models**:  
- **BERT** — Semantic understanding  
- **TF-IDF** — Keyword importance  
- **N-gram overlap** — Phrase matching  
- **Levenshtein distance** — Character-level edits  

Generates **detailed reports**, **visualizations**, and **CSV/JSON exports** — all saved in **unique, timestamped folders** to prevent overwrites.  
Fully **cross-platform** (Windows, macOS, Linux) and designed for **teachers and educators**.

---

## Features

| Feature | Description |
|--------|-------------|
| **4 AI Models** | BERT, TF-IDF, N-gram, Levenshtein |
| **Weighted Scoring** | Fully customizable via `--weights` |
| **Per-file & Pair Reports** | Clear breakdown of similarity |
| **Visualizations** | Bar chart + Similarity Heatmap (PNG) |
| **Export Options** | CSV / JSON |
| **Safe Reports** | Timestamped + auto-counter (`_1`, `_2`, etc.) |
| **File Support** | `.txt`, `.md`, `.docx` |
| **Cross-Platform** | Windows, macOS, Linux |

---

## Folder Structure

```
PlagiGuard/
├── plag_check.py   ← Main script
├── install.py                       ← One-click installer
├── install.sh                       ← Auto-created (optional)
├── reports/                         ← Auto-generated
└── your_submissions/                ← Your folder to scan
```

> **Important**: Keep `plag_check.py` and `install.py` in the **same folder**.

---

## Installation

## Get the Code from GitHub

```bash
# 1. Clone the repository
git clone https://github.com/KamaleshRanganathan/Project_PlagiGuard.git

# 2. Enter the project folder
cd Project_PlagiGuard

# 3. Install everything (packages + NLTK data)
python install.py

```
For Step 3 run in bash
**Or double-click `install.py`** (Windows/macOS)

---

## Usage

### Step 1: Place your documents in a folder

```
your_submissions/
├── s1.txt
├── s2.txt
├── s3.docx
└── ...
```

### Step 2: Run PlagiGuard

```bash
python plag_check.py your_submissions/ --export report.csv --visualize
```

### Available Options

| Flag | Description |
|------|-----------|
| `--mode full` | Show per-file + pairs (default) |
| `--mode pairs` | Show only suspicious pairs |
| `--threshold 0.7` | Set similarity threshold (0.0–1.0) |
| `--weights '{"bert":0.6,"tfidf":0.2,"ngram":0.1,"lev":0.1}'` | Custom model weights |
| `--export report.csv` | Export results to CSV |
| `--export data.json` | Export results to JSON |
| `--visualize` | Generate PNG charts |
| `--no-tables` | Plain text output (no tabulate) |

---

## Example Output

```
reports/
└── report_20251102_2230_1/
    ├── per_file_scores.png
    ├── similarity_heatmap.png
    └── report.csv
```

---

## One-Click Script (Optional)

```bash
chmod +x install.sh
./install.sh
```

---

## Requirements

- **Python 3.8 or higher**
- Internet connection (first run: downloads BERT model ~90 MB)

---

## Final Folder to Share

```
PlagiGuard/
├── plag_check.py
├── install.py
├── README.md
└── your_submissions/   ← Add your files here
```

> **Zip this folder** → Share via email, USB, or Google Drive  
> **Works on any computer** — no setup needed

---

## Made for Teachers

> **No coding. No setup.**  
> Just drag, run, and review.

---

**PlagiGuard** — *Because originality matters.*  
*Built with care for educators.*

--- 

**Ready for GitHub, email, or classroom deployment.**
