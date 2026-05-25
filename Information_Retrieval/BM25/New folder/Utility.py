from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
import numpy as np
import json

stop_words = set(stopwords.words("english")) #assuming english lexicon (all other dropped)

class BM25:
    """
    Okapi BM25
    """
    def __init__(self, index, k1=1.5, b=0.75):
        self.postings = index["postings"]
        self.doc_len = index["doc_len"]
        self.N = index["N"]
        self.avgdl = index["avgdl"]
        self.k1 = k1
        self.b = b

    def Idf(self, n_qt):
        return np.log(1 + (self.N / n_qt))

    def Tf(self, tf, dl):
        return tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))

    def formula(self, query_terms):
        """
            scores: defaultdict(float) to accumulate BM25 scores for each document
            postings: dictionary of {doc_id: term frequency} for the current term
            TF: term frequency that depends on document length
            IDF: inverse document frequency of the term
            n_q: number of documents containing the current term
        """
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
    
class BM25PR(BM25):
    """
    Pseudo-relevance feedback.
    """

    def prf_weights(self, relevant_docs, df, eps=1e-6):
        postings = self.postings
        N = self.N
        v = len(relevant_docs)
        if v == 0:
            return {}

        v_i = Counter()
        for d in relevant_docs:
            for t in df.iloc[d]["tokens"]:
                if d in postings.get(t, {}):
                    v_i[t] += 1

        weights = {}
        for t, vi in v_i.items():
            df_i = len(postings[t])
            p_i = (vi + 0.5) / (v + 1)
            u_i = (df_i - vi + 0.5) / (N - v + 1)

            ratio1 = (p_i / (1 - p_i))
            ratio2 = ((1 - u_i) / u_i)
            ratio1 = max(ratio1, eps)  # ensure non 0
            ratio2 = max(ratio2, eps) # ensure non 0
            w = np.log(ratio1) + np.log(ratio2)

            if w > 0:
                weights[t] = w

        return weights

    def score_rf(self, weights, seen_docs=set()):
        scores = defaultdict(float)
        for term, w in weights.items():
            postings = self.postings.get(term, {})
            for doc_id in postings:
                if doc_id in seen_docs:
                    continue
                scores[doc_id] += w
        return dict(scores)


def cleanup(df):
    """
    Ensure 'tokens' column is properly loaded.
    """
    df["tokens"] = df["tokens"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
    return df

def tokenize(text):
    """
    tokenization.
    """
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in stop_words]

def inverted_index(df):
    """
    Constructs inverted index for future use/storage
    """
    postings = defaultdict(dict)
    doc_len = {}

    for doc_id, tokens in enumerate(df["tokens"]):
        doc_len[doc_id] = len(tokens)

        for term in tokens:
            postings[term][doc_id] = postings[term].get(doc_id, 0) + 1
    
    N = len(df)
    avgdl = sum(doc_len.values()) / N

    return {
        "postings": dict(postings),
        "doc_len": doc_len,
        "N": N,
        "avgdl": avgdl
    }