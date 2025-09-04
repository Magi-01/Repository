import pandas as pd
import re
import math
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle

# ----------------- Setup -----------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
BM25_STATE_FILE = "bm25.pkl"

def normalize_query(query: str) -> str:
    tokens = query.lower().split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

def preprocess(text):
    """Preprocess text: lowercase, remove non-alphanum, stem, remove stopwords."""
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text.lower())
    tokens = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return tokens

# ----------------- BM25 Class -----------------
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = [preprocess(doc) for doc in docs]
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        self.doc_len = [len(d) for d in self.docs]
        self.index = defaultdict(dict)
        self.build_index()

        # Vocabulary and document frequencies
        self.vocab = set()
        self.doc_freqs = defaultdict(int)
        for term, postings in self.index.items():
            self.vocab.add(term)
            self.doc_freqs[term] = len(postings)
        self.feedback = {}

    def build_index(self):
        for doc_id, doc in enumerate(self.docs):
            freqs = Counter(doc)
            for term, f in freqs.items():
                self.index[term][doc_id] = f

    def idf(self, term):
        df = len(self.index.get(term, {}))
        return 0 if df == 0 else math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens, doc_id):
        score = 0.0
        doc_len = self.doc_len[doc_id]
        for term in query_tokens:
            f = self.index.get(term, {}).get(doc_id, 0)
            if f == 0:
                continue
            idf = self.idf(term)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        return score

    def search(self, query, top_k=10):
        query_tokens = preprocess(query)
        scores = [(doc_id, self.score(query_tokens, doc_id)) for doc_id in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# ----------------- Persistence -----------------
def save_bm25_state(bm25, filepath=BM25_STATE_FILE):
    with open(filepath, "wb") as f:
        pickle.dump(bm25, f)

def load_bm25_state(filepath=BM25_STATE_FILE):
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# ----------------- Feedback -----------------
def save_feedback_in_bm25(bm25, query, expanded_terms, relevant_docs):
    norm_query = normalize_query(query)
    if norm_query in bm25.feedback:
        bm25.feedback[norm_query]["expanded_terms"] = list(
            set(bm25.feedback[norm_query]["expanded_terms"] + expanded_terms)
        )
        bm25.feedback[norm_query]["relevant_docs"] = list(
            set(bm25.feedback[norm_query]["relevant_docs"] + relevant_docs)
        )
    else:
        bm25.feedback[norm_query] = {
            "expanded_terms": expanded_terms,
            "relevant_docs": relevant_docs
        }

def augment_query_with_feedback_from_bm25(bm25, query):
    norm_query = normalize_query(query)
    if norm_query in bm25.feedback:
        expanded_terms = bm25.feedback[norm_query]["expanded_terms"]
        return query + " " + " ".join(expanded_terms)
    return query

# ----------------- Evaluation -----------------
def precision_at_k(relevant_docs, retrieved_docs, k=10):
    retrieved_k = [doc for doc, _ in retrieved_docs[:k]]
    return len(set(retrieved_k) & set(relevant_docs)) / k

def recall_at_k(relevant_docs, retrieved_docs, k=10):
    retrieved_k = [doc for doc, _ in retrieved_docs[:k]]
    return len(set(retrieved_k) & set(relevant_docs)) / len(relevant_docs)

def average_precision(relevant_docs, retrieved_docs, k=10):
    retrieved_k = [doc for doc, _ in retrieved_docs[:k]]
    hits = 0
    sum_prec = 0
    for i, doc_id in enumerate(retrieved_k, start=1):
        if doc_id in relevant_docs:
            hits += 1
            sum_prec += hits / i
    return sum_prec / hits if hits > 0 else 0

def evaluate_all_queries(test_queries, bm25, df, top_k=10):
    all_precisions, all_recalls, all_average_precisions, all_f1, all_ndcg, all_mrr = [], [], [], [], [], []
    per_query_metrics = {}

    def dcg(rels):
        return sum((2**r - 1) / math.log2(idx + 2) for idx, r in enumerate(rels))

    for query_text, relevant_docs in test_queries.items():
        results = bm25.search(query_text, top_k=top_k)
        retrieved_ids = [doc_id for doc_id, _ in results]

        rel_list = [1 if doc in relevant_docs else 0 for doc in retrieved_ids]

        prec = precision_at_k(relevant_docs, results, k=top_k)
        rec = recall_at_k(relevant_docs, results, k=top_k)
        ap = average_precision(relevant_docs, results, k=top_k)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        ideal_rels = sorted([1]*len(relevant_docs) + [0]*(top_k - len(relevant_docs)), reverse=True)[:top_k]
        ndcg = dcg(rel_list) / dcg(ideal_rels) if dcg(ideal_rels) > 0 else 0

        mrr = 0
        for rank, rel in enumerate(rel_list, start=1):
            if rel:
                mrr = 1 / rank
                break

        per_query_metrics[query_text] = {
            f'Precision@{top_k}': prec,
            f'Recall@{top_k}': rec,
            'AveragePrecision': ap,
            f'F1@{top_k}': f1,
            f'nDCG@{top_k}': ndcg,
            f'MRR@{top_k}': mrr
        }

        all_precisions.append(prec)
        all_recalls.append(rec)
        all_average_precisions.append(ap)
        all_f1.append(f1)
        all_ndcg.append(ndcg)
        all_mrr.append(mrr)

    mean_metrics = {
        f'MeanPrecision@{top_k}': sum(all_precisions)/len(all_precisions),
        f'MeanRecall@{top_k}': sum(all_recalls)/len(all_recalls),
        'MAP': sum(all_average_precisions)/len(all_average_precisions),
        f'MeanF1@{top_k}': sum(all_f1)/len(all_f1),
        f'MeannDCG@{top_k}': sum(all_ndcg)/len(all_ndcg),
        f'MeanMRR@{top_k}': sum(all_mrr)/len(all_mrr)
    }

    return per_query_metrics, mean_metrics

# ----------------- Pseudo-Relevance Feedback -----------------
def pseudo_relevance_feedback_iterative(bm25, query, relevant_doc_ids, n_iterations=3, expansion_terms=5):
    """
    Iterative pseudo-relevance feedback using standard BM25 scoring.
    
    Args:
        bm25: BM25 object
        query: str, current query
        relevant_doc_ids: list of int, initial relevant document IDs
        n_iterations: int, number of PRF iterations
        expansion_terms: int, number of terms to add per iteration

    Returns:
        final_query: str, query after expansion
        all_expanded_terms: list of str, all learned expansion terms
        final_relevant_docs: list of int, updated relevant document IDs
    """
    current_query = query
    all_expanded_terms = []

    for _ in range(n_iterations):
        # Collect term frequencies from relevant docs
        term_counter = Counter()
        for doc_id in relevant_doc_ids:
            term_counter.update(bm25.docs[doc_id])

        # Remove original query terms from expansion
        original_tokens = preprocess(current_query)
        for t in original_tokens:
            term_counter.pop(t, None)

        # Select top expansion terms
        top_terms = [t for t, _ in term_counter.most_common(expansion_terms)]
        if not top_terms:
            break

        all_expanded_terms.extend(top_terms)
        current_query += " " + " ".join(top_terms)

        # Update relevant docs using top-k BM25 results
        search_results = bm25.search(current_query, top_k=10)
        relevant_doc_ids = [doc_id for doc_id, _ in search_results[:5]]

    return current_query, all_expanded_terms, relevant_doc_ids


if __name__ == "__main__":
    # Load actual dataset
    df = pd.read_csv("Information Retrieval\\openalex_papers.csv").fillna('').reset_index(drop=True)
    abstracts = df['abstract_text'].tolist()

    # Load or build BM25
    bm25 = load_bm25_state("bm25_with_feedback1.pk2")
    if bm25 is None:
        bm25 = BM25(abstracts)
        save_bm25_state(bm25, "bm25_with_feedback1.pk2")
    
    test_queries = {
        # AI general
        "What are the main approaches in artificial intelligence?": [1, 2, 4],
        "How does modern AI differ from classical AI methods?": [1, 2],
        "Which AI topics are covered in lecture notes?": [3],
        "What were the key outcomes of the 19th IJCAI conference?": [5],
        "How can AI improve performance in medicine?": [6, 10],
        "What is the link between integer programming and AI?": [7],
        "How is AI applied in radiology?": [8],
        "What are multi-agent systems and their role in distributed AI?": [9],
        "How has AI evolved in healthcare over time?": [10],
        # Neural networks & deep learning
        "What is the significance of the ImageNet dataset in deep learning?": [11, 13],
        "How are deep convolutional neural networks used for image classification?": [11, 13],
        "What is the foundational theory behind neural networks?": [12, 15],
        "How does dropout help prevent overfitting in neural networks?": [14],
        "How can neural networks reduce the dimensionality of data?": [17],
        "What are key trends in deep learning research?": [18],
        "How is knowledge distilled in a neural network?": [19],
        "How do sequence-to-sequence models work in neural networks?": [20],
        "What are the practical applications of neural networks for pattern recognition?": [15, 16],
        # Supervised and semi-supervised learning
        "What are the fastest algorithms for supervised learning?": [22, 23],
        "How does semi-supervised learning differ from supervised learning?": [21, 22, 25, 27, 29],
        "What literature exists for semi-supervised learning?": [22, 24],
        "How can Gaussian fields be used for semi-supervised learning?": [25],
        "What is the ALBERT model used for in language representations?": [26],
        "How can fairness be achieved in supervised learning?": [27],
        "What methods exist for virtual adversarial training in semi-supervised learning?": [29],
        "Which introduction resources cover semi-supervised learning?": [28],
        "What surveys provide a comprehensive view of semi-supervised learning?": [29],
        # Unsupervised learning
        "What are common techniques for unsupervised learning?": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
        "How can jigsaw puzzles be used for unsupervised visual representation learning?": [31],
        "How can unsupervised learning estimate depth and ego-motion from video?": [35],
        "What is the role of convolutional deep belief networks in unsupervised learning?": [36],
        "How can slow feature analysis extract invariances in unsupervised learning?": [41],
        "What is deep clustering and how does it help unsupervised learning?": [38],
        "How can LSTMs be applied to unsupervised video representation learning?": [39],
        # AI & privacy / ethics
        "What are the main cybersecurity risks posed by generative AI like ChatGPT?": [42],
        "How can AI ethics and privacy be ensured in big data research?": [43],
        "What taxonomy exists for AI-related privacy risks like deepfakes?": [44],
        "How does federated learning preserve privacy in AI systems?": [45],
        "What privacy challenges arise in smart contracts using AI?": [46],
        "How do US and Chinese opinions differ on AI privacy?": [47],
        "What are patient perceptions of privacy in AI-driven healthcare?": [48],
        "How do AI-driven chatbots impact user experience and privacy perception?": [49],
        "What ethical and privacy challenges exist in healthcare AI?": [48, 50],
        "How do AI technologies relate to privacy and security concerns?": [50],
        # Combined or broader questions
        "What are the overlaps between AI in healthcare and AI ethics?": [6, 10, 43, 48, 50],
        "How do neural networks contribute to deep learning and AI research?": [12, 15, 18, 19, 20],
        "What methods integrate semi-supervised and unsupervised learning?": [21, 25, 29, 30, 32],
        "What are the latest advances in AI research and publications?": [5, 18, 31, 42],
        "How can AI improve performance in both medicine and security?": [6, 42, 45, 48, 50],
        "What are the challenges in training neural networks efficiently?": [14, 17, 19, 22],
        "How is visual feature learning approached in unsupervised methods?": [31, 36, 38, 39],
        "What role does conference research play in AI advancements?": [3, 5],
        "How can fairness and opportunity be evaluated in AI systems?": [27, 43, 48],
        "Which techniques are used to reduce overfitting and improve model generalization?": [14, 19, 22, 29],
        # Practical AI applications
        "How is AI applied in multiagent distributed systems?": [9],
        "What are future directions for integer programming and AI integration?": [7],
        "How is AI applied in medical imaging and radiology?": [6, 8, 10],
        "What approaches improve AI performance in pattern recognition tasks?": [15, 16, 20],
        "How can AI models balance privacy, security, and user experience?": [42, 45, 46, 49, 50],
    }

    # Interactive loop for user queries
    for query, relevant_docs in test_queries.items():

        # Augment query with PRF feedback stored in BM25
        query_aug = augment_query_with_feedback_from_bm25(bm25, query)

        # Retrieve and display results
        results = bm25.search(query_aug, top_k=10)  # <-- use search(), not search_fuzzy
        print(f"\nTop results for query: '{query_aug}'\n")
        for rank, (doc_id, score) in enumerate(results, start=1):
            snippet = " ".join(df.loc[doc_id, 'abstract_text'].split()[:30])
            print(f"{rank}. Doc ID: {doc_id}, Score: {score:.4f}")
            print(f"   {snippet}...\n")

        # Use pre-defined relevant documents to update PRF
        selected = relevant_docs
        if selected:
            try:
                learned_terms = pseudo_relevance_feedback_iterative(
                    bm25, query_aug, relevant_doc_ids=selected, n_iterations=3, expansion_terms=5
                )[1]  # Only get learned_terms

                save_feedback_in_bm25(bm25, query, learned_terms, selected)
                save_bm25_state(bm25, "bm25_with_feedback1.pk2")
                print(f"Feedback saved for query '{query}'.\n")
            except Exception as e:
                print(f"Invalid input, feedback not saved. Error: {e}\n")
                
    # Evaluate all queries
    per_query_metrics, mean_metrics = evaluate_all_queries(test_queries, bm25, df, top_k=10)
    print("\n=== Evaluation Metrics ===")
    print(per_query_metrics)
    print(mean_metrics)
