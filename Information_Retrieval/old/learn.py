import pandas as pd
import pickle
from collections import defaultdict
from query import BM25Fuzzy, preprocess_fuzzy, augment_query_with_feedback_from_bm25  # import your BM25 class and utils

# ----------------- Load BM25 and Data -----------------
BM25_PICKLE = "C:\\Users\\mutua\\Documents\\Repository\\Repository\\bm25_with_feedback1.pkl"
DATA_CSV = "C:\\Users\\mutua\\Documents\\Repository\\Repository\\Information Retrieval\\openalex_papers.csv"

# Load saved BM25
with open(BM25_PICKLE, "rb") as f:
    bm25 = pickle.load(f)

# Load dataset
df = pd.read_csv(DATA_CSV).fillna('').reset_index(drop=True)

# ----------------- Interactive Query Loop -----------------
def main():
    print("BM25 Inference Mode. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        # Augment query with saved PRF feedback
        query_aug = augment_query_with_feedback_from_bm25(bm25, query)

        # Search
        results = bm25.search_fuzzy(query_aug, top_k=10)

        # Display results
        print(f"\nTop results for query: '{query_aug}'\n")
        if not results:
            print("No documents found.")
            continue

        for rank, (doc_id, score) in enumerate(results, start=1):
            if doc_id >= len(df):
                continue
            snippet = " ".join(df.loc[doc_id, 'abstract_text'].split()[:30])
            title = df.loc[doc_id, 'title']
            print(f"{rank}. {title} (Doc ID: {doc_id}, Score: {score:.4f})")
            print(f"   {snippet}...\n")

if __name__ == "__main__":
    main()
