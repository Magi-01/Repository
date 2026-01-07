import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

stop_words = set(stopwords.words("english"))

def tokenize(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in stop_words]

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("/home/fadhla/Documents/School/Repository/Information Retrieval/openalex_papers4.csv").fillna('').reset_index(drop=True)
df["tokens"] = df["abstract_text"].apply(tokenize)
doc_term_counts = df["tokens"].apply(Counter)
N = len(df)
avgdl = df["tokens"].str.len().mean()

# ----------------------------
# BM25 class
# ----------------------------
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def score(self, query_terms, doc_counts, doc_len):
        scores = []
        for i, doc in enumerate(doc_counts):
            score = 0
            for term in query_terms:
                if term in doc:
                    n_t = sum(term in d for d in doc_counts)
                    idf = np.log((N - n_t + 0.5) / (n_t + 0.5))
                    tf = doc[term]
                    denom = tf + self.k1 * (1 - self.b + self.b * (doc_len[i] / avgdl))
                    score += idf * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return np.array(scores)

bm25 = BM25()

# ----------------------------
# RSJ / Relevance Feedback Functions
# ----------------------------
def compute_rsj_weights(query_terms, rel_docs, nonrel_docs, doc_term_counts):
    R = len(rel_docs)
    N_total = len(rel_docs) + len(nonrel_docs)
    r_t = {t: sum(t in doc_term_counts.iloc[d] for d in rel_docs) for t in query_terms}
    n_t = {t: sum(t in doc_term_counts.iloc[d] for d in nonrel_docs) for t in query_terms}
    weights = {}
    for term in query_terms:
        weights[term] = np.log(
            (r_t[term] + 0.5) * (N_total - n_t[term] - R + r_t[term] + 0.5) /
            ((R - r_t[term] + 0.5) * (n_t[term] + 0.5))
        )
    return weights

def score_with_rsj(query_terms, doc_term_counts, term_weights):
    scores = np.zeros(len(doc_term_counts))
    for i, doc in enumerate(doc_term_counts):
        for t in query_terms:
            if t in doc:
                scores[i] += term_weights.get(t, 0)
    return scores

# ----------------------------
# Pseudo-Relevance Feedback
# ----------------------------
def pseudo_feedback(query_terms, doc_term_counts, initial_scores, top_k=10):
    ranked = np.argsort(initial_scores)[::-1]
    rel_docs = ranked[:top_k]
    nonrel_docs = ranked[top_k:]
    return compute_rsj_weights(query_terms, rel_docs, nonrel_docs, doc_term_counts)

# ----------------------------
# Clustering / Unsupervised RF
# ----------------------------
corpus_texts = df["abstract_text"].tolist()
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(corpus_texts)
kmeans = KMeans(n_clusters=20, random_state=42).fit(X)

def cluster_feedback(query_terms, doc_term_counts, clusters, cluster_labels):
    query_vec = vectorizer.transform([" ".join(query_terms)])
    cluster_distances = np.linalg.norm(clusters.cluster_centers_ - query_vec.toarray(), axis=1)
    closest_cluster = np.argmin(cluster_distances)
    rel_docs = np.where(cluster_labels == closest_cluster)[0]
    nonrel_docs = np.where(cluster_labels != closest_cluster)[0]
    return compute_rsj_weights(query_terms, rel_docs, nonrel_docs, doc_term_counts)

# ----------------------------
# Evaluation Metrics
# ----------------------------
def reciprocal_rank(ranked_indices, correct_index):
    rank_pos = np.where(ranked_indices == correct_index)[0][0] + 1
    return 1.0 / rank_pos

def compute_metrics(rr_list, top_k=10):
    rr_array = np.array(rr_list)
    mrr = np.mean(rr_array)
    hit = np.sum(rr_array >= 1/top_k) / len(rr_array)
    return hit, mrr

# ----------------------------
# Main Evaluation Loop
# ----------------------------
TOP_K = 10
results = {"BM25": [], "User_RF": [], "PRF": [], "Cluster_RF": []}

# Simulate a query set; here using a fraction of df
query_df, _ = train_test_split(df, test_size=0.8, random_state=42)

# Example supervised relevance: assume first 3 documents per query are relevant
for i, query_row in query_df.iterrows():
    query_terms = tokenize(query_row["title"])
    if not query_terms:
        continue
    correct_index = df.index.get_loc(query_row.name)
    
    # Vanilla BM25
    scores_bm25 = bm25.score(query_terms, doc_term_counts, df["tokens"].str.len().values)
    ranked_bm25 = np.argsort(scores_bm25)[::-1]
    results["BM25"].append(reciprocal_rank(ranked_bm25, correct_index))
    
    # User RF (supervised) - here we simulate feedback
    rel_docs = [correct_index]  # only the true document is relevant
    nonrel_docs = [i for i in range(N) if i != correct_index]
    term_weights_user = compute_rsj_weights(query_terms, rel_docs, nonrel_docs, doc_term_counts)
    scores_user_rf = score_with_rsj(query_terms, doc_term_counts, term_weights_user)
    ranked_user_rf = np.argsort(scores_user_rf)[::-1]
    results["User_RF"].append(reciprocal_rank(ranked_user_rf, correct_index))
    
    # PRF
    term_weights_prf = pseudo_feedback(query_terms, doc_term_counts, scores_bm25, top_k=TOP_K)
    scores_prf = score_with_rsj(query_terms, doc_term_counts, term_weights_prf)
    ranked_prf = np.argsort(scores_prf)[::-1]
    results["PRF"].append(reciprocal_rank(ranked_prf, correct_index))
    
    # Cluster RF
    term_weights_cluster = cluster_feedback(query_terms, doc_term_counts, kmeans, kmeans.labels_)
    scores_cluster = score_with_rsj(query_terms, doc_term_counts, term_weights_cluster)
    ranked_cluster = np.argsort(scores_cluster)[::-1]
    results["Cluster_RF"].append(reciprocal_rank(ranked_cluster, correct_index))

# ----------------------------
# Compute final metrics
# ----------------------------
metrics_summary = {}
for method, rr_list in results.items():
    hit, mrr = compute_metrics(rr_list)
    metrics_summary[method] = {"Hit@10": hit, "MRR": mrr}

# Display as table
import pprint
pprint.pprint(metrics_summary)

methods = list(metrics_summary.keys())
hit_values = [metrics_summary[m]["Hit@10"] for m in methods]
mrr_values = [metrics_summary[m]["MRR"] for m in methods]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x - width/2, hit_values, width, label='Hit@10')
rects2 = ax.bar(x + width/2, mrr_values, width, label='MRR')

ax.set_ylabel('Scores')
ax.set_title('IR Method Comparison')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0,1)
ax.legend()

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')
plt.show()
