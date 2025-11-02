#!/usr/bin/env python3
"""
PlagiGuard - AI Plagiarism Detector
Reports saved in BASE_DIR/reports/report_YYYYMMDD_HHMM/
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
import string

# Optional visualization libs
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Required libs
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from Levenshtein import ratio
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError as e:
    print(f"Error: Missing lib {e}. Run: python install.py")
    sys.exit(1)

# Optional docx support
try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global BERT model
BERT_MODEL = None

# Defaults
DEFAULT_WEIGHTS = {"bert": 0.5, "tfidf": 0.2, "ngram": 0.2, "lev": 0.1}
SIMILARITY_THRESHOLD = 0.70
MAX_LEV_LEN = 2000

# === BASE DIRECTORY (where this script lives) ===
BASE_DIR = Path(__file__).parent.resolve()
REPORTS_ROOT = BASE_DIR / "reports"

def get_report_dir() -> Path:
    """Create unique timestamped report folder with counter suffix if needed."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = f"report_{timestamp}"
    report_dir = REPORTS_ROOT / base_name

    # If folder exists, append _1, _2, etc.
    counter = 1
    while report_dir.exists():
        report_dir = REPORTS_ROOT / f"{base_name}_{counter}"
        counter += 1

    report_dir.mkdir(parents=True, exist_ok=False)
    logger.info(f"Report folder: {report_dir}")
    return report_dir

def load_bert_model():
    global BERT_MODEL
    if BERT_MODEL is None:
        logger.info("Loading Sentence-BERT model...")
        BERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return BERT_MODEL

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(cleaned)

def get_ngrams(text: str, n: int) -> set:
    words = text.split()
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}

def compute_bert_similarity(docs: List[str]) -> np.ndarray:
    n = len(docs)
    if n < 2:
        return np.zeros((n, n))
    model = load_bert_model()
    embeds = model.encode(docs, show_progress_bar=False)
    sim = cosine_similarity(embeds)
    np.fill_diagonal(sim, 0.0)
    return sim

def compute_ngram_overlap(docs: List[str], n_gram: int = 3) -> np.ndarray:
    n = len(docs)
    if n < 2:
        return np.zeros((n, n))
    ngrams_list = [get_ngrams(doc, n_gram) for doc in docs]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(ngrams_list[i] & ngrams_list[j])
            union = len(ngrams_list[i] | ngrams_list[j])
            score = inter / union if union else 0.0
            sim[i, j] = sim[j, i] = score
    return sim

def compute_levenshtein_similarity(docs: List[str]) -> np.ndarray:
    n = len(docs)
    if n < 2 or any(len(d) > MAX_LEV_LEN for d in docs):
        logger.warning(f"Skipping Levenshtein: Docs too long (> {MAX_LEV_LEN} chars).")
        return np.zeros((n, n))
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            score = ratio(docs[i], docs[j])
            sim[i, j] = sim[j, i] = score
    return sim

def compute_tfidf_similarity(docs: List[str]) -> np.ndarray:
    n = len(docs)
    if n < 2:
        return np.zeros((n, n))
    try:
        vec = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf = vec.fit_transform(docs)
        sim = cosine_similarity(tfidf)
        np.fill_diagonal(sim, 0.0)
        return sim
    except ValueError:
        logger.warning("TF-IDF failed: Empty/short docs.")
        return np.zeros((n, n))

def detect_plagiarism(raw_docs: List[str], proc_docs: List[str], filenames: List[str],
                      weights: Dict[str, float], threshold: float) -> Tuple[List[Dict], List[Dict]]:
    n = len(raw_docs)
    if n < 2:
        return [], []

    logger.info(f"Processing {n} docs ({n*(n-1)//2} pairs)...")

    # Compute matrices
    logger.info("1/4 TF-IDF...")
    tfidf_mat = compute_tfidf_similarity(proc_docs)
    logger.info("2/4 N-Gram...")
    ngram_mat = compute_ngram_overlap(proc_docs)
    logger.info("3/4 Levenshtein...")
    lev_mat = compute_levenshtein_similarity(raw_docs)
    logger.info("4/4 BERT...")
    bert_mat = compute_bert_similarity(raw_docs)

    # Weighted combined matrix
    combined_mat = (
        bert_mat * weights["bert"] +
        tfidf_mat * weights["tfidf"] +
        ngram_mat * weights["ngram"] +
        lev_mat * weights["lev"]
    )

    def get_max_excl_self(mat):
        row = mat[i].copy()
        row[i] = -1
        return np.max(row).item(), int(np.argmax(row))

    per_file = []
    for i in range(n):
        bert_score, bert_pair_idx = get_max_excl_self(bert_mat)
        tfidf_score, tfidf_pair_idx = get_max_excl_self(tfidf_mat)
        ngram_score, ngram_pair_idx = get_max_excl_self(ngram_mat)
        lev_score, lev_pair_idx = get_max_excl_self(lev_mat)

        row_comb = combined_mat[i].copy()
        row_comb[i] = -1
        combined_max = np.max(row_comb).item()

        per_file.append({
            "file": filenames[i],
            "combined": combined_max,
            "bert": {"score": bert_score, "pair": filenames[bert_pair_idx]},
            "tfidf": {"score": tfidf_score, "pair": filenames[tfidf_pair_idx]},
            "ngram": {"score": ngram_score, "pair": filenames[ngram_pair_idx]},
            "lev": {"score": lev_score, "pair": filenames[lev_pair_idx]}
        })

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = combined_mat[i, j]
            if score >= threshold:
                pairs.append({
                    "file1": filenames[i],
                    "file2": filenames[j],
                    "score": float(score)
                })

    per_file.sort(key=lambda x: x["combined"], reverse=True)
    pairs.sort(key=lambda x: x["score"], reverse=True)
    return per_file, pairs

def visualize_results(per_file: List[Dict], pairs: List[Dict], filenames: List[str], out_dir: Path):
    if not HAS_MPL:
        return

    # Bar chart
    files_short = [f[:20] + "..." if len(f) > 20 else f for f in [r["file"] for r in per_file]]
    scores = [r["combined"] for r in per_file]
    plt.figure(figsize=(10, 6))
    plt.barh(files_short, scores, color='skyblue')
    plt.xlabel('Combined Similarity Score')
    plt.title('Per-File Max Similarity')
    plt.tight_layout()
    plt.savefig(out_dir / "per_file_scores.png", dpi=150)
    plt.close()
    logger.info("Saved: per_file_scores.png")

    # Heatmap
    n = len(filenames)
    mat = np.zeros((n, n))
    for p in pairs:
        i = filenames.index(p["file1"])
        j = filenames.index(p["file2"])
        mat[i, j] = mat[j, i] = p["score"]
    plt.figure(figsize=(10, 8))
    plt.imshow(mat, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Similarity')
    plt.xticks(range(n), [f[:10] for f in filenames], rotation=45)
    plt.yticks(range(n), [f[:10] for f in filenames])
    plt.title('Similarity Heatmap')
    plt.tight_layout()
    plt.savefig(out_dir / "similarity_heatmap.png", dpi=150)
    plt.close()
    logger.info("Saved: similarity_heatmap.png")

def print_report(per_file: List[Dict], pairs: List[Dict], mode: str, threshold: float, use_tabulate: bool = True):
    if mode in ["full", "0"]:
        print("\n=== PER-FILE BREAKDOWN ===")
        table_data = []
        for r in per_file:
            row = [r["file"], f"{r['combined']:.1%}"]
            for k in ["bert", "tfidf", "ngram", "lev"]:
                row.append(f"{r[k]['score']:.1%} ({r[k]['pair'][:20]})")
            table_data.append(row)
        headers = ["File", "Combined", "BERT", "TF-IDF", "N-Gram", "Lev"]
        if HAS_TABULATE and use_tabulate:
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            print(" | ".join(headers))
            for row in table_data:
                print(" | ".join(row))

    print(f"\n=== SUSPICIOUS PAIRS (>= {threshold:.0%}) ===")
    if not pairs:
        print("No suspicious pairs found.")
        return
    print(f"Found {len(pairs)} pairs.")
    table_data = [[p["file1"][:20], p["file2"][:20], f"{p['score']:.1%}"] for p in pairs]
    headers = ["File1", "File2", "Score"]
    if HAS_TABULATE and use_tabulate:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print(" | ".join(headers))
        for row in table_data:
            print(" | ".join(row))

def export_reports(per_file: List[Dict], pairs: List[Dict], export_path: str, report_dir: Path):
    if not export_path:
        return
    path = report_dir / Path(export_path).name  # Save inside report folder
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, np.float32):
                return float(obj)
            return obj
        data = {"per_file": convert(per_file), "pairs": convert(pairs)}
        with path.open("w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported JSON: {path.name}")

    elif path.suffix == ".csv":
        import csv
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Type", "File1", "File2", "Score", "Details"])
            for r in per_file:
                details = {k: r[k]["score"] for k in ["bert", "tfidf", "ngram", "lev"]}
                writer.writerow([
                    "per_file", r["file"], "", f"{r['combined']:.4f}",
                    json.dumps(details, ensure_ascii=False)
                ])
            for p in pairs:
                writer.writerow(["pair", p["file1"], p["file2"], f"{p['score']:.4f}", ""])
        logger.info(f"Exported CSV: {path.name}")

def setup_nltk():
    required = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    for pkg in required:
        try:
            nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}' if pkg == 'wordnet' else f'corpus/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)
    logger.info("NLTK ready.")

def load_documents(folder: Path) -> Tuple[List[str], List[str]]:
    docs, names = [], []
    for p in sorted(folder.glob("*.txt")) + sorted(folder.glob("*.md")):
        names.append(p.name)
        docs.append(p.read_text(encoding="utf-8"))
    if HAS_DOCX:
        for p in sorted(folder.glob("*.docx")):
            doc = Document(p)
            text = "\n".join(para.text for para in doc.paragraphs)
            names.append(p.name)
            docs.append(text)
    return docs, names

def main():
    print("PlagiGuard v1.0 - AI Plagiarism Detector")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="PlagiGuard")
    parser.add_argument("folder", type=Path, help="Folder with documents")
    parser.add_argument("--mode", choices=["full", "pairs", "0", "1"], default="full")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--weights", type=str, default=json.dumps(DEFAULT_WEIGHTS))
    parser.add_argument("--export", type=str, help="Export: report.csv or data.json")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--no-tables", action="store_true")
    args = parser.parse_args()

    if not args.folder.is_dir():
        logger.error(f"Folder not found: {args.folder}")
        sys.exit(1)

    setup_nltk()
    raw_docs, filenames = load_documents(args.folder)
    if len(raw_docs) < 2:
        logger.error("Need >=2 documents.")
        sys.exit(1)

    logger.info("Preprocessing...")
    proc_docs = [preprocess_text(d) for d in raw_docs]

    try:
        weights = json.loads(args.weights)
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    except Exception as e:
        logger.error(f"Invalid weights: {e}")
        sys.exit(1)

    per_file, pairs = detect_plagiarism(raw_docs, proc_docs, filenames, weights, args.threshold)

    # Create timestamped report folder in BASE DIR
    report_dir = get_report_dir()

    if args.visualize:
        visualize_results(per_file, pairs, filenames, report_dir)

    mode = "pairs" if args.mode in ["pairs", "1"] else "full"
    print_report(per_file, pairs, mode, args.threshold, not args.no_tables)

    if args.export:
        export_reports(per_file, pairs, args.export, report_dir)

    print(f"\nReport saved in: {report_dir}")

if __name__ == "__main__":
    main()