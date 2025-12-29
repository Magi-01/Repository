import pandas as pd
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import ast

from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/fadhla/Documents/School/Repository/Information Retrieval/openalex_papers4.csv").fillna('').reset_index(drop=True)

class BM25:
    def __init__(self, frq, sd, avgwdl, k1, b, N, n_qt):
        """
            frq: Frequency of word in document D,
            sd: size of document (in words),
            avgwdl: average documents length in corpus(/collection),
            k1: term frequency scaling,
            b: document length normalization,
            N: Total documents,
            n_qt: Number of documents containing query term
        """

        self.frq = frq
        self.sd = sd
        self.avgwdl = avgwdl
        self.k1 = k1
        self.b = b
        self.N = N
        self.n_qt = n_qt

    def Idf(self):
        upper = self.N - self.n_qt + 0.5
        lower = self.n_qt + 0.5
        return np.log(upper / lower)

    def Tf(self):
        upper = self.frq
        lower = self.frq + self.k1 * (
            1 - self.b + self.b * (self.sd / self.avgwdl)
        )
        return upper / lower


    def formula(self):
        return np.sum(self.Idf() * self.Tf(), axis=0, dtype=np.float64)

#______________________________________________#
stop_words = set(stopwords.words("english"))

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

df["concepts_list"] = df["concepts"].apply(parse_concepts)

def tokenize(text):
    tokens = word_tokenize(text.lower())
    return [
        t for t in tokens
        if t.isalpha() and t not in stop_words
    ]

df["tokens"] = df["abstract_text"].fillna("").apply(tokenize)

df["sd"] = df["tokens"].apply(len)

sd = df["sd"].values.reshape(1, -1)   # (1, n_docs)
avgwdl = df["sd"].mean()
N = len(df)

"""
if __name__ == '__main__':
    query = input("What is your Query? \n")
    query_terms = tokenize(query)

    concept = input("Do you want to include concepts? \n")
    concept_terms = tokenize(concept) if concept else None

    doc_term_counts = df["tokens"].apply(Counter)

    frq = np.array([
        [doc.get(term, 0) for doc in doc_term_counts]
        for term in query_terms
    ])

    n_qt = np.array([
        sum(term in doc for doc in doc_term_counts)
        for term in query_terms
    ]).reshape(-1, 1)

    bm25 = BM25(
        frq=frq,
        sd=sd,
        avgwdl=avgwdl,
        k1=1.5,
        b=0.75,
        N=N,
        n_qt=n_qt
    )

    scores = bm25.formula()    # (n_docs,)
    df["bm25_score"] = scores

    alpha = 0
    if concept_terms:
        df["concept_score"] = df["concepts_list"].apply(lambda x: concept_overlap(concept_terms, x))
        alpha = 0.3
    else:
        df["concept_score"] = 1  # no boost

    # final score
    df["final_score"] = df["bm25_score"] * (1 + alpha * df["concept_score"])

    df = df.sort_values("final_score", ascending=False)
    pd.set_option('display.max_colwidth', None)
    print(df.loc[df["final_score"] > 0, ["title", "year", "final_score"]].head(10))
    print(df.loc[df["final_score"] > 0, ["title", "year", "final_score"]].shape[0]," documents found out of ", df.shape[0])
"""

if __name__ == "__main__":
    precision_hits = 0
    recall_hits = 0
    reciprocal_ranks = []
    top_k = 10

    doc_term_counts = df["tokens"].apply(Counter)
    query_df, _ = train_test_split(df, test_size=0.2, random_state=42)

    for i, query_row in query_df.iterrows():
        query_terms = query_row["tokens"]
        query_concepts = query_row.get("concepts_list", [])

        frq = np.array([[doc.get(term, 0) for doc in doc_term_counts] for term in query_terms])
        n_qt = np.array([sum(term in doc for doc in doc_term_counts) for term in query_terms]).reshape(-1,1)

        bm25 = BM25(frq, sd, avgwdl, k1=1.5, b=0.75, N=N, n_qt=n_qt)
        scores = bm25.formula()

        # Concept boost
        concept_scores = df["concepts_list"].apply(lambda x: concept_overlap(query_concepts, x))
        final_scores = scores * (1 + 0 * concept_scores)

        # Rank documents
        ranked_indices = np.argsort(final_scores)[::-1]

        # Find rank of the correct document
        correct_index = df.index.get_loc(query_row.name)
        rank_position = np.where(ranked_indices == correct_index)[0][0] + 1  # 1-based

        # Metrics
        if rank_position <= top_k:
            precision_hits += 1
            recall_hits += 1  # only one relevant doc
        reciprocal_ranks.append(1 / rank_position)

    # Compute final metrics
    precision_at_k = precision_hits / len(query_df)
    recall_at_k = recall_hits / len(query_df)
    mrr = np.mean(reciprocal_ranks)

    print(f"Precision@{top_k}: {precision_at_k:.4f}")
    print(f"Recall@{top_k}: {recall_at_k:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")