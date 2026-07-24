import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from BM25 import compute_BM25PR
import json
import os
from Utility import inverted_index

Query = [
    # Computer Science / AI
    "machine learning", "artificial intelligence", "neural networks", "deep learning",
    "computer vision", "reinforcement learning", "natural language processing", "AI ethics",
    "robotics", "knowledge graphs",
    # Physics
    "quantum mechanics", "climate modeling", "string theory", "particle physics", "astrophysics",
    "condensed matter physics", "gravitational waves", "thermodynamics", "optics", "plasma physics",
    # Biology / Medicine
    "genome sequencing", "cancer immunotherapy", "CRISPR gene editing", "stem cell therapy",
    "epigenetics", "microbiome research", "protein folding", "neuroscience", "vaccine development",
    "bioinformatics",
    # Social Sciences
    "behavioral economics", "urban sociology", "political polarization", "education policy",
    "social networks", "gender studies", "migration studies", "organizational behavior",
    "public health policy", "criminology",
    # Humanities
    "medieval literature", "renaissance art", "philosophy of mind", "linguistics",
    "cultural anthropology", "classical archaeology", "music theory", "modern literature",
    "history of science", "religious studies"
]

# ----------------------
# Helper functions
# ----------------------
def load_labels(label_path):
    if not os.path.exists(label_path):
        return {}
    with open(label_path, "r") as f:
        return json.load(f)

def precision_at_k(ranked_docs, relevant_docs, k):
    ranked_topk = ranked_docs[:k]
    return sum([1 for doc in ranked_topk if doc in relevant_docs]) / k

def recall_at_k(ranked_docs, relevant_docs, k):
    ranked_topk = ranked_docs[:k]
    return sum([1 for doc in ranked_topk if doc in relevant_docs]) / len(relevant_docs) if relevant_docs else 0.0

def mean_average_precision(ranked_docs, relevant_docs, k):
    # MAP@k
    relevant_set = set(relevant_docs)
    hits = 0
    sum_precisions = 0
    for i, doc_id in enumerate(ranked_docs[:k], 1):
        if doc_id in relevant_set:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / min(len(relevant_docs), k) if relevant_docs else 0.0

def mean_average_recall(ranked_docs, relevant_docs, k):
    return recall_at_k(ranked_docs, relevant_docs, k)

# ----------------------
# Main evaluation loop
# ----------------------
CSV_PATH = "./openalex10.csv"
LABEL_PATH = "./label.json"
TOP_K = [1, 5, 10, 100, 1000]
MAX_ITER = 10
STABILITY = 0.9

# Load dataset
df = pd.read_csv(CSV_PATH).fillna("").reset_index(drop=True)
df["tokens"] = df["tokens"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
invert_idx = inverted_index(df)  # inverted index here

for k in TOP_K:
    map_scores = []
    mar_scores = []
    queries = []
    if os.path.exists(LABEL_PATH):
        os.remove(LABEL_PATH)
    label_data = load_labels(LABEL_PATH)
    for q in tqdm(Query):
        relevant_docs = set(label_data.get(q, {}).keys())
        if not relevant_docs:
            # treat top-k from BM25PR as pseudo-ground-truth
            df_scores = compute_BM25PR(df, q, invert_idx, LABEL_PATH,
                                    top_k=k, stability_threshold=STABILITY,
                                    max_iter=MAX_ITER, verbose=False)
            relevant_docs = set(df_scores["id"].tolist())

        df_scores = compute_BM25PR(df, q, invert_idx, LABEL_PATH,
                                top_k=k, stability_threshold=STABILITY,
                                max_iter=MAX_ITER, verbose=False)
        
        ranked_docs = df_scores["id"].tolist()

        map_k = mean_average_precision(ranked_docs, relevant_docs, k)
        mar_k = mean_average_recall(ranked_docs, relevant_docs, k)

        map_scores.append(map_k)
        mar_scores.append(mar_k)
        queries.append(q)

    # ----------------------
    # Plotting
    # ----------------------
    plt.figure(figsize=(14,6))
    plt.plot(range(len(queries)), map_scores, marker='o', label=f"MAP@{k}")
    plt.plot(range(len(queries)), mar_scores, marker='x', label=f"MAR@{k}")
    plt.xticks(range(len(queries)), queries, rotation=90)
    plt.xlabel("Queries")
    plt.ylabel("Score")
    plt.title(f"BM25+PR Evaluation (k={k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"bm25pr_evaluation_{k}.png")

#plt.show()
