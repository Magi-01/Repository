import pickle
import json
import os
from collections import defaultdict


# Label Storage Utilities

def save_labels(labels, query, label_path):
    all_labels = load_labels(label_path)
    query_labels = all_labels.get(query, {})
    for doc_id, rel in labels.items():
        query_labels[str(doc_id)] = int(rel)
    all_labels[query] = query_labels
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        for q, labs in all_labels.items():
            f.write(json.dumps({q: labs}) + "\n")


def load_labels(label_path):
    if not os.path.exists(label_path):
        return {}
    all_labels = {}
    with open(label_path, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                all_labels.update(record)
            except json.JSONDecodeError:
                continue
    return all_labels


def get_labels_for_query(query, label_path):
    all_labels = load_labels(label_path)
    query_labels = all_labels.get(query, {})
    seen_docs = set(map(int, query_labels.keys()))
    relevant_docs = {int(doc_id) for doc_id, rel in query_labels.items() if rel == 1}
    return seen_docs, relevant_docs


# Inverted index storage

def inverted_index(df):
    postings = defaultdict(dict)
    doc_len = {}
    for doc_id, tokens in enumerate(df["tokens"]):
        doc_len[doc_id] = len(tokens)
        term_counts = defaultdict(int)
        for term in tokens:
            term_counts[term] += 1
        for term, tf in term_counts.items():
            postings[term][doc_id] = tf
    N = len(df)
    avgdl = sum(doc_len.values()) / N if N > 0 else 0
    postings = {term: dict(docs) for term, docs in postings.items()}
    return {"postings": postings, "doc_len": doc_len, "N": N, "avgdl": avgdl}

def save_index(index, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Inverted index saved to {file_path}")

def load_index(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Inverted index file not found: {file_path}")
    with open(file_path, "rb") as f:
        index = pickle.load(f)
    return index