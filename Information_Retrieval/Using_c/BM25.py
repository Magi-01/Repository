import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import ast
import ctypes
import os
from nltk import PorterStemmer, WordNetLemmatizer

stop_words = set(stopwords.words("english"))

#_______________________________________________________________________#

def parse_concepts(concepts):
    if pd.isna(concepts) or concepts == '':
        return []
    try:
        return ast.literal_eval(concepts)
    except:
        return [concepts]
    
def concept_overlap(query_concepts, doc_concepts):
    if not query_concepts or not doc_concepts:
        return 0
    return len(set(query_concepts) & set(doc_concepts)) / len(set(query_concepts))

def tokenize(text):
    tokens = word_tokenize(text.lower())
    return [
        t for t in tokens
        if t.isalpha() and t not in stop_words
    ]

#________________________________________________________________________#

df = pd.read_csv("openalex_papers4.csv").fillna('').reset_index(drop=True)
df["concepts_list"] = df["concepts"].apply(parse_concepts)

df["tokens"] = df["abstract_text"].fillna("").apply(tokenize)

df["sd"] = df["tokens"].apply(len)
vocab = sorted(set(term for tokens in df["tokens"] for term in tokens))

query_df, _ = train_test_split(df, test_size=0.2, random_state=42)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(r"C:\TDM-GCC-64\bin")
# Load shared library
lib_path = os.path.abspath("./BM25.dll")
lib = ctypes.CDLL(os.path.normpath(lib_path))

# Define argtypes
lib.bm25_batch_multiquery.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # doc_term_matrix
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # query_term_matrix
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # idf
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # doc_lengths
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")  # scores
]

# Prepare data
D = len(df)
V = len(vocab)  # full vocabulary size
Q = len(query_df)

doc_term_matrix = np.zeros((D, V), dtype=np.int32)
term_to_index = {term: i for i, term in enumerate(vocab)}
for d, tokens in enumerate(df["tokens"]):
    for t in tokens:
        idx = term_to_index[t]
        doc_term_matrix[d, idx] += 1

query_term_matrix = np.zeros((Q, V), dtype=np.int32)
for q, tokens in enumerate(query_df["tokens"]):
    for t in tokens:
        if t in term_to_index:
            query_term_matrix[q, term_to_index[t]] += 1

# Fill your doc_term_matrix and query_term_matrix here...
doc_lengths = np.array(df["sd"], dtype=np.float64)
avgdl = doc_lengths.mean()
k1 = 1.5
b = 0.75

idf = np.zeros(V, dtype=np.float64)
scores = np.zeros((Q, D), dtype=np.float64)

lib.compute_idf.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # doc_term_matrix
    ctypes.c_int,  # D
    ctypes.c_int,  # V
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")  # idf
]

lib.compute_idf.restype = None  # void function

D = doc_term_matrix.shape[0]
V = doc_term_matrix.shape[1]

# Compute IDF
doc_term_matrix = np.ascontiguousarray(doc_term_matrix, dtype=np.int32)
idf = np.ascontiguousarray(idf, dtype=np.float64)
lib.compute_idf(doc_term_matrix, D, V, idf)

# Run batched BM25
lib.bm25_batch_multiquery(doc_term_matrix, query_term_matrix, idf, doc_lengths,
                           D, V, Q, avgdl, k1, b, scores)

# scores.shape should be (Q, D)
print("Scores shape:", scores.shape)
print("Sample scores (first query, first 10 docs):", scores[0, :10])

top_k = 10
for q_idx, query_row in enumerate(query_df.itertuples()):
    # Get the scores for this query
    q_scores = scores[q_idx, :]
    
    # Rank documents descending
    ranked_indices = np.argsort(q_scores)[::-1]  # indices of top scoring documents
    
    # Optional: print top titles
    print(f"\nQuery {q_idx} top {top_k} documents:")
    for rank, doc_idx in enumerate(ranked_indices[:top_k], start=1):
        title = df.iloc[doc_idx]["title"]
        score = q_scores[doc_idx]
        print(f"{rank}. {title} (score={score:.4f})")

precision_hits = 0
recall_hits = 0
reciprocal_ranks = []

for q_idx, query_row in enumerate(query_df.itertuples()):
    q_scores = scores[q_idx, :]
    ranked_indices = np.argsort(q_scores)[::-1]

    # Correct document is the query itself (assuming leave-one-out)
    correct_index = df.index.get_loc(query_row.Index)
    rank_position = np.where(ranked_indices == correct_index)[0][0] + 1  # 1-based

    top_k = 10
    if rank_position <= top_k:
        precision_hits += 1
        recall_hits += 1
    reciprocal_ranks.append(1 / rank_position)

precision_at_k = precision_hits / Q
recall_at_k = recall_hits / Q
mrr = np.mean(reciprocal_ranks)

print(f"Precision@{top_k}: {precision_at_k:.4f}")
print(f"Recall@{top_k}: {recall_at_k:.4f}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")