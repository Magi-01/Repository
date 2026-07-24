import pandas as pd
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import ast
import pickle
import os
import matplotlib.pyplot as plt
import math
import json
import datetime
import matplotlib.pyplot as plt
from itertools import product

from sklearn.model_selection import train_test_split
import heapq
INDEX_PATH = "/home/fadhla/Documents/School/Repository/Information_Retrieval/inv_index.pkl"
# Take data from csv and construct a dataframe, sanity check for empty rows
df = pd.read_csv("/home/fadhla/Documents/School/Repository/Information_Retrieval/openalex10.csv").fillna('').reset_index(drop=True)
df.head(10)
# Common stop symbols in the english lexicon
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
class BM25:
    def __init__(self, index, k1, b):
        """
            The following are numpy array/vectors
            frq: Frequency of word in document D,
            sd: size of document (in words),
            avgwdl: average documents length in corpus(/collection),
            k1: term frequency scaling,
            b: document length normalization,
            N: Total documents,
            n_qt: Number of documents containing query term
        """

        self.postings = index["postings"]
        self.doc_len  = index["doc_len"]
        self.N        = index["N"]
        self.avgdl    = index["avgdl"]
        self.k1 = k1
        self.b = b

    def Idf(self, n_qt):
        return np.log(1 + (self.N / n_qt))

    def Tf(self, tf, dl):
        return tf*(self.k1 + 1) / (
            tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        )


    def formula(self, query_terms):
        scores = defaultdict(float)

        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue

            n_q = len(postings)
            idf = self.Idf(n_q)

            for doc_id, tf in postings.items():
                dl = self.doc_len[doc_id]
                scores[doc_id] += idf * self.Tf(tf, dl)

        return dict(scores)
    
def tokenize(text):
    tokens = word_tokenize(text.lower())
    return [
        t for t in tokens
        if t.isalpha() and t not in stop_words
    ]

def inverted_index(df):
    postings = defaultdict(dict)
    doc_len = {}

    for doc_id, tokens in enumerate(df["tokens"]):
        doc_len[doc_id] = len(tokens)

        for term in tokens:
            postings[term][doc_id] = postings[term].get(doc_id, 0) + 1
    
    N = len(df)
    avgwdl = sum(doc_len.values()) / N

    return {
        "postings": dict(postings),
        "doc_len": doc_len,
        "N": N,
        "avgdl": avgwdl
    }

def save_index(index, file_path = "/home/fadhla/Documents/School/Repository/Information_Retrieval/inv_index.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(index, f)

def load_index(file_path):
    with open(file_path, "rb") as f:
        index = pickle.load(f)
    return index

df["tokens"] = df["tokens"].apply(ast.literal_eval)

if os.path.exists(INDEX_PATH):
    invert_idx = load_index(INDEX_PATH)
else:
    invert_idx = inverted_index(df)
    save_index(invert_idx, INDEX_PATH)
query = "kernel ridge regression"

query_terms = tokenize(query)

bm25 = BM25(invert_idx, k1=1.5, b=0.75)
scores = bm25.formula(query_terms)
df["score"] = df.index.map(scores).fillna(0)
df = df.sort_values("score", ascending=False)
pd.set_option('display.max_colwidth', None)

print(df.loc[df["score"] > 0, ["title", "year", "score"]].head(10))


print(df.loc[df["score"] > 0, ["title", "year", "score"]].shape[0]," documents found out of ", df.shape[0])

LABEL_PATH = "pseudo_labels.json"

def preprocess_text(text):
    """Tokenize, lowercase, lemmatize, stem, remove stopwords."""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

# --- Pseudo-Relevance Feedback ---
class BM25PRF:
    def __init__(self, index, k1=1.5, b=0.75):
        self.postings = index["postings"]
        self.doc_len = index["doc_len"]
        self.N = index["N"]
        self.avgdl = index["avgdl"]
        self.k1 = k1
        self.b = b

    # Standard BM25
    def Idf(self, n_qt):
        return math.log(1 + (self.N / n_qt))

    def Tf(self, tf, dl):
        return tf*(self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))

    def formula(self, query_terms):
        scores = defaultdict(float)
        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            n_q = len(postings)
            idf = self.Idf(n_q)
            for doc_id, tf in postings.items():
                dl = self.doc_len[doc_id]
                scores[doc_id] += idf * self.Tf(tf, dl)
        return dict(scores)

    # -------------------------
    # Pseudo-Relevance Feedback
    # -------------------------
    def prf_weights(self, relevant_docs, df):
        """
        RSJ weights:
        p_i = (|v_i| + 0.5)/(v + 1)
        u_i = (df_i - |v_i| + 0.5)/(N - v + 1)
        """
        postings = self.postings
        N = self.N
        v = len(relevant_docs)
        if v == 0:
            return {}

        v_i = Counter()
        for d in relevant_docs:
            for t in df.loc[d, "tokens"]:
                if d in postings.get(t, {}):
                    v_i[t] += 1

        weights = {}
        for t, vi in v_i.items():
            df_i = len(postings[t])
            p_i = (vi + 0.5) / (v + 1)
            u_i = (df_i - vi + 0.5) / (N - v + 1)
            try:
                w = math.log(1 + (p_i * (1 - u_i)) / (u_i * (1 - p_i)))
                if w > 0:
                    weights[t] = w
            except ValueError:
                continue
        return weights

    def score_rf(self, weights, seen_docs, decay_docs=None, decay_factor=0.7):
        """
        Scores docs with optional decay for previous top documents
        """
        scores = defaultdict(float)
        for term, w in weights.items():
            for doc_id, tf in self.postings.get(term, {}).items():
                if doc_id in seen_docs and (not decay_docs or doc_id not in decay_docs):
                    continue
                score_add = w
                if decay_docs and doc_id in decay_docs:
                    score_add *= decay_factor
                scores[doc_id] += score_add
        return dict(scores)

# -------------------------
# Pseudo-label storage
# -------------------------
def save_labels(labels, query):
    """
    Save labels as: {"query": {doc_id: relevance}}
    """
    record = {query: {str(doc_id): int(rel) for doc_id, rel in labels.items()}}
    with open(LABEL_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

def load_labels():
    """
    Load all pseudo-labels.
    Returns: dict query -> {doc_id: relevance}
    """
    if not os.path.exists(LABEL_PATH):
        return {}
    all_labels = {}
    with open(LABEL_PATH, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                all_labels.update(record)
            except json.JSONDecodeError:
                continue
    return all_labels

def get_labels_for_query(query):
    all_labels = load_labels()
    query_labels = all_labels.get(query, {})
    seen_docs = set(map(int, query_labels.keys()))
    relevant_docs = set(int(doc_id) for doc_id, rel in query_labels.items() if rel == 1)
    return query_labels, seen_docs, relevant_docs

# -------------------------
# PRF Loop
# -------------------------
def pseudo_relevance_loop(query, bm25rf, df, top_k=100, stability_threshold=0.7,
                          max_iter=100, decay_factor=0.5):
    query_labels, seen_docs, relevant_docs = get_labels_for_query(query)
    non_relevant_docs = seen_docs - relevant_docs

    # If no previous relevant docs, initialize with BM25 top-k
    if not relevant_docs:
        print("No previous relevant docs, using first top_k BM25 docs as pseudo-relevant...")
        initial_scores = bm25rf.formula(tokenize(query))
        ranked = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = ranked[:top_k]
        for doc_id, _ in top_docs:
            seen_docs.add(doc_id)
            relevant_docs.add(doc_id)
        save_labels({doc_id:1 for doc_id, _ in top_docs}, query)

    prev_top_terms = set()
    running_query_terms = set(preprocess_text(query))

    for iteration in range(max_iter):
        # --- Compute PRF weights ---
        weights = bm25rf.prf_weights(relevant_docs, df)
        if not weights:
            print(f"[Iteration {iteration}] No relevant docs — stopping.")
            break

        # --- Score remaining docs (decay previous top docs) ---
        scores = bm25rf.score_rf(weights, seen_docs, decay_docs=seen_docs, decay_factor=decay_factor)
        if not scores:
            print(f"[Iteration {iteration}] No more retrievable documents.")
            break

        # --- Top-k documents ---
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = ranked_docs[:top_k]
        top_doc_ids = set(doc_id for doc_id, _ in top_docs)

        # --- Update running sets ---
        new_labels = {doc_id: 1 for doc_id in top_doc_ids}
        for doc_id in new_labels:
            seen_docs.add(doc_id)
            relevant_docs.add(doc_id)

        # --- Update query expansion terms ---
        top_terms = set()
        for doc_id in top_doc_ids:
            text = df.loc[doc_id, "abstract_text"]
            top_terms.update(preprocess_text(text))
        running_query_terms.update(top_terms)

        # --- Compute precision, recall ---
        retrieved = len(new_labels)
        retrieved_relevant = len(top_doc_ids & relevant_docs)
        precision = retrieved_relevant / retrieved if retrieved > 0 else 0
        recall = retrieved_relevant / len(relevant_docs) if relevant_docs else 0

        # --- Convergence check using term Jaccard ---
        overlap_ratio = 0
        if prev_top_terms:
            overlap_ratio = len(top_terms & prev_top_terms) / len(top_terms | prev_top_terms)

        print(f"[Iteration {iteration}] Precision@{top_k}: {precision:.4f}, Recall: {recall:.4f}")
        print(f"[Iteration {iteration}] Top-{top_k} term Jaccard overlap: {overlap_ratio:.4f}")
        print(f"[Iteration {iteration}] Expanded query (sample 50 terms): {list(running_query_terms)[:50]}")
        print(f"[Iteration {iteration}] Top doc IDs: {list(top_doc_ids)[:10]}\n")

        # --- Save pseudo-labels ---
        save_labels(new_labels, query)

        # --- Convergence check ---
        if overlap_ratio >= stability_threshold:
            print(f"Converged at iteration {iteration} (Jaccard >= {stability_threshold})")
            break

        prev_top_terms = top_terms

    return relevant_docs, running_query_terms

def run_prf_cross_validation(query, bm25rf, df, invert_idx, 
                             top_k_list=[10,100,1000], 
                             stability_thresholds=[0.7,0.8,0.9],
                             max_iter=50, decay_factor=0, debug=True):
    """
    Runs PRF with cross-validation for top_k and stability_threshold.
    Keeps previous documents in scoring with decay.
    Expands query using lemmatization + stemming.
    Saves graphs for each run.
    """
    results = []

    for top_k, stability_threshold in product(top_k_list, stability_thresholds):
        if debug:
            print(f"\n--- Running top_k={top_k}, stability_threshold={stability_threshold} ---\n")
        
        # Load previous labels
        _, seen_docs, relevant_docs = get_labels_for_query(query)
        cumulative_scores = defaultdict(float)
        prev_top_docs = set()
        
        # Initialize running query
        running_query_terms = tokenize(query)

        iteration_metrics = []

        for iteration in range(max_iter):
            # Compute PRF weights using current relevant docs
            weights = bm25rf.prf_weights(relevant_docs, df)
            if not weights:
                if debug: print(f"[Iteration {iteration}] No relevant docs — stopping.")
                break

            # Score remaining documents with decay
            scores = bm25rf.score_rf(weights, seen_docs)
            for doc_id, score in scores.items():
                cumulative_scores[doc_id] = decay_factor * cumulative_scores.get(doc_id, 0) + score

            if not cumulative_scores:
                if debug: print(f"[Iteration {iteration}] No more retrievable documents.")
                break

            # Pick top-k cumulative scores
            ranked_docs = sorted(cumulative_scores.items(), key=lambda x: x[1], reverse=True)
            top_docs = ranked_docs[:top_k]
            top_doc_ids = set(doc_id for doc_id, _ in top_docs)

            # Update seen and relevant docs
            for doc_id in top_doc_ids:
                seen_docs.add(doc_id)
                relevant_docs.add(doc_id)

            # Compute precision, recall
            retrieved = len(top_doc_ids)
            retrieved_relevant = len(top_doc_ids & relevant_docs)
            precision = retrieved_relevant / retrieved if retrieved > 0 else 0
            recall = retrieved_relevant / len(relevant_docs) if relevant_docs else 0

            # Jaccard overlap with previous top-k
            jaccard = len(top_doc_ids & prev_top_docs) / len(top_doc_ids | prev_top_docs) if prev_top_docs else 0.0

            if debug:
                print(f"[Iteration {iteration}] Precision@{top_k}: {precision:.4f}, Recall: {recall:.4f}, Jaccard: {jaccard:.4f}")
            
            iteration_metrics.append({
                "iteration": iteration,
                "precision": precision,
                "recall": recall,
                "jaccard": jaccard,
                "relevant_docs": len(relevant_docs)
            })

            # --- Check convergence ---
            if jaccard >= stability_threshold:
                if debug: print(f"Converged at iteration {iteration} (Jaccard >= {stability_threshold})\n")
                break

            prev_top_docs = top_doc_ids

            # --- Expand running query ---
            top_terms = []
            for d in top_doc_ids:
                tokens = df.loc[d, "tokens"]
                for t in tokens:
                    lemma = lemmatizer.lemmatize(t)
                    stem = stemmer.stem(lemma)
                    top_terms.append(stem)
            running_query_terms = list(set(running_query_terms) | set(top_terms))

        # Save results
        results.append({
            "top_k": top_k,
            "stability_threshold": stability_threshold,
            "metrics": iteration_metrics
        })

        # --- Plot metrics over iterations ---
        iterations = [m["iteration"] for m in iteration_metrics]
        precisions = [m["precision"] for m in iteration_metrics]
        recalls = [m["recall"] for m in iteration_metrics]
        jaccards = [m["jaccard"] for m in iteration_metrics]

        plt.figure(figsize=(10,6))
        plt.plot(iterations, precisions, label="Precision")
        plt.plot(iterations, recalls, label="Recall")
        plt.plot(iterations, jaccards, label="Jaccard")
        plt.xlabel("Iteration")
        plt.ylabel("Metric")
        plt.title(f"PRF Metrics (top_k={top_k}, threshold={stability_threshold})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"prf_metrics_topk{top_k}_th{stability_threshold}.png")
        plt.close()

        # --- Precision-Recall curve ---
        plt.figure(figsize=(6,6))
        plt.plot(recalls, precisions, marker='o')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (top_k={top_k}, threshold={stability_threshold})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"prf_precision_recall_topk{top_k}_th{stability_threshold}.png")
        plt.close()

    return results

# -------------------------
# Example usage
# -------------------------
bm25rf = BM25PRF(invert_idx)
query = "regression"
thresholds = [0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9]
results = run_prf_cross_validation(query, bm25rf, df, invert_idx=invert_idx, stability_thresholds=thresholds)
#relevant_docs = pseudo_relevance_loop(query, bm25rf, df, top_k=100, stability_threshold=0.9, max_iter=100)