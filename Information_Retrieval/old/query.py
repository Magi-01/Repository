import pandas as pd
import re
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix
from time import time
import json
import os
import statistics

# ----------------- Setup -----------------
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ----------------- Load CSV -----------------
# Get current directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(base_dir, "openalex_papers.csv")).fillna('')
docs = [(str(row['title']) + " " + str(row['abstract_text'])) for _, row in df.iterrows()]

# -------- Load Ground Truth --------
json_path = os.path.join(base_dir, "ground_values.json")
with open(json_path, "r") as f:
    ground_data = json.load(f)

test_queries = ground_data["ground_values"]

# ----------------- Preprocessing -----------------
def preprocess(text):
    if not text:  # empty string or None
        return []
    if isinstance(text, list):
        text = " ".join(text)
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    bigrams = ['_'.join([tokens[i], tokens[i+1]]) for i in range(len(tokens)-1)]
    return tokens + bigrams


# ----------------- Vectorized BM25 -----------------
class BM25Vectorized:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs_tokens = [preprocess(doc) for doc in docs]
        self.N = len(docs)
        self.avgdl = np.mean([len(d) for d in self.docs_tokens])
        # Build vocabulary and doc-term matrix
        self.vocab = {}
        self.doc_freq = Counter()
        data, rows, cols = [], [], []
        for doc_id, tokens in enumerate(self.docs_tokens):
            freqs = Counter(tokens)
            for term, f in freqs.items():
                if term not in self.vocab:
                    self.vocab[term] = len(self.vocab)
                idx = self.vocab[term]
                data.append(f)
                rows.append(doc_id)
                cols.append(idx)
                self.doc_freq[term] += 1
        self.DTM = csr_matrix((data, (rows, cols)), shape=(self.N, len(self.vocab)), dtype=float)
        self.idf = np.array([np.log((self.N - self.doc_freq[t] + 0.5) / (self.doc_freq[t] + 0.5) + 1) for t in self.vocab])
        self.doc_len = np.array([len(tokens) for tokens in self.docs_tokens])

    def search(self, query, top_k=10):
        q_tokens = preprocess(query)
        q_indices = [self.vocab[t] for t in q_tokens if t in self.vocab]
        if not q_indices:
            return []
        # BM25 formula
        scores = np.zeros(self.N)
        for idx in q_indices:
            f = self.DTM[:, idx].toarray().flatten()
            scores += self.idf[idx] * ((f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)))
        top_docs = np.argsort(-scores)[:top_k]
        return [(int(doc_id), float(scores[doc_id])) for doc_id in top_docs]

# ----------------- PRF -----------------
def pseudo_relevance_feedback(bm25, query, top_k=5, expansion_terms=5):
    results = bm25.search(query, top_k=top_k)
    top_doc_ids = [doc_id for doc_id, _ in results]
    term_counter = Counter()
    for doc_id in top_doc_ids:
        term_counter.update(bm25.docs_tokens[doc_id])
    # Remove original query tokens
    for t in preprocess(query):
        term_counter.pop(t, None)
    top_terms = [t for t, _ in term_counter.most_common(expansion_terms)]
    expanded_query = query + " " + " ".join(top_terms)
    return expanded_query, top_terms

# ----------------- Metrics -----------------
def precision_at_k(relevant, retrieved, k=10):
    retrieved_k = [doc for doc, _ in retrieved[:k]]
    return len(set(retrieved_k) & set(relevant)) / k

def recall_at_k(relevant, retrieved, k=10):
    retrieved_k = [doc for doc, _ in retrieved[:k]]
    return len(set(retrieved_k) & set(relevant)) / max(len(relevant), 1)

def average_precision(relevant, retrieved, k=10):
    retrieved_k = [doc for doc, _ in retrieved[:k]]
    hits, sum_prec = 0, 0
    for i, doc_id in enumerate(retrieved_k, 1):
        if doc_id in relevant:
            hits += 1
            sum_prec += hits / i
    return sum_prec / hits if hits > 0 else 0

def dcg(rels):
    return sum((2**r - 1)/np.log2(i+2) for i, r in enumerate(rels))

def evaluate_query(query, relevant_docs, bm25):
    results = bm25.search(query, top_k=10)
    rel_list = [1 if doc_id in relevant_docs else 0 for doc_id, _ in results]
    prec = precision_at_k(relevant_docs, results)
    rec = recall_at_k(relevant_docs, results)
    ap = average_precision(relevant_docs, results)
    ndcg_score = dcg(rel_list)/dcg(sorted(rel_list, reverse=True)) if dcg(rel_list) > 0 else 0
    mrr = next((1/(i+1) for i, r in enumerate(rel_list) if r), 0)
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    return {
        'query': query,
        'Precision@10': prec,
        'Recall@10': rec,
        'AveragePrecision': ap,
        'F1@10': f1,
        'nDCG@10': ndcg_score,
        'MRR@10': mrr
    }

# -------- Latency Measurement --------
def measure_latency(bm25, queries, top_k=5, runs=100):
    times = []
    for i in range(runs):
        query = queries[i % len(queries)]  # cycle through queries
        start = time()
        bm25.search(query, top_k=top_k)
        end = time()
        times.append(end - start)
    avg_time = statistics.mean(times)
    stdev_time = statistics.stdev(times)
    return avg_time, stdev_time


bm25 = BM25Vectorized(docs)

# ----------------- Query -----------------
user_query = input("Insert Query: ")
result = None

start_prf = time()
expanded_query, learned_terms = pseudo_relevance_feedback(bm25, user_query)
prf_time = time() - start_prf

# Build a mapping from OpenAlex ID to BM25 doc index
id_to_index = {row['id']: idx for idx, row in df.iterrows()}
numeric_id_to_url = {i: row['id'] for i, row in df.iterrows()}


# ----------------- Latency Measure -----------------
sample_queries = list(test_queries.keys())[:-1]  # take 10 test queries
avg, stdev = measure_latency(bm25, sample_queries, runs=100)

print(f"Average query latency over 100 runs: {avg:.6f} sec (±{stdev:.6f})")

# Initialize accumulators
precision_list = []
recall_list = []
ap_list = []
f1_list = []
ndcg_list = []
mrr_list = []

# Loop through sample queries and evaluate
for query in sample_queries:
    relevant_docs_urls = [numeric_id_to_url[num] for num in test_queries[query] if num in numeric_id_to_url]
    bm25_relevant = [id_to_index[url] for url in relevant_docs_urls]
    res = evaluate_query(query, bm25_relevant, bm25)
    
    precision_list.append(res['Precision@10'])
    recall_list.append(res['Recall@10'])
    ap_list.append(res['AveragePrecision'])
    f1_list.append(res['F1@10'])
    ndcg_list.append(res['nDCG@10'])
    mrr_list.append(res['MRR@10'])

# Compute averages
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_ap = np.mean(ap_list)
avg_f1 = np.mean(f1_list)
avg_ndcg = np.mean(ndcg_list)
avg_mrr = np.mean(mrr_list)

print(f"Average metrics over {len(test_queries.keys())} queries:")
print(f"Precision@10: {avg_precision:.4f}")
print(f"Recall@10: {avg_recall:.4f}")
print(f"AveragePrecision: {avg_ap:.4f}")
print(f"F1@10: {avg_f1:.4f}")
print(f"nDCG@10: {avg_ndcg:.4f}")
print(f"MRR@10: {avg_mrr:.4f}")


if user_query in test_queries:
    relevant_docs = test_queries[user_query]
    numeric_id_to_url = {idx: row['id'] for idx, row in df.iterrows()}

    relevant_docs_urls = [numeric_id_to_url[num] for num in relevant_docs if num in numeric_id_to_url]
    bm25_relevant = [id_to_index[url] for url in relevant_docs_urls]

    result = evaluate_query(user_query, relevant_docs, bm25)
    print(result)
else:
    print("No ground truth available; skipping metrics.")


if result:
    for lab, res in result.items():
        print(f"\n{lab}:{res}")


print(f"PRF expansion time: {prf_time:.4f} seconds")
print("\nOriginal Query:", user_query)
print("Expanded Query:", expanded_query)
print("Learned Terms:", learned_terms)
if result is not None:
    print(", ".join(f"{k}:{r}" for k, r in result.items()))
    
print("\nTop Results:")
for rank, (doc_id, score) in enumerate(bm25.search(expanded_query, top_k=5), 1):
    snippet = " ".join(df.loc[doc_id, 'abstract_text'].split()[:30])
    print(f"{rank}. Paper ID: {df.loc[doc_id, 'id']}, Score: {score:.4f}")
    print(f"   Title: {df.loc[doc_id, 'title']}")
    print(f"   Snippet: {snippet}...\n")


# Or if you have an expanded query from PRF
results = bm25.search(expanded_query, top_k=10)
print("Top retrieved doc IDs and scores:")
for rank, (doc_id, score) in enumerate(results, 1):
    print(f"{rank}: Doc ID {doc_id}, Score {score:.4f}")