import pandas as pd
import os
import argparse
from collections import Counter
import time
from Utility import BM25, BM25PR, cleanup, tokenize, inverted_index
from save_load import save_index, load_index, save_labels, get_labels_for_query
import numpy as np

# --- Pseudo-Relevance Feedback ---

def pseudo_relevance_loop(query, bm25rf, df, label_path, top_k=10, 
                                    precision_threshold=0.9, max_iter=20, verbose=True):
    """
    Iterative pseudo-relevance feedback using precision top-k convergence.
    Measured as P = #current top-k relevant to previous top-k
    Initial top-k documents is determined user query

    verbose: If True, debugs.
    """
    # Load or initialize labels
    seen_docs, relevant_docs = get_labels_for_query(query, label_path)

    # If there exists an instance of labeling for query, then use that
    if relevant_docs:
        if verbose:
            print(f"[INFO] Query '{query}' already has pseudo-relevant labels ({len(relevant_docs)} docs). Skipping PR iterations.")
        # Compute weights from existing relevant docs
        weights = bm25rf.prf_weights(relevant_docs, df)
        return weights

    # If no relevant docs yet, use first BM25 top-k as pseudo-relevant
    if not relevant_docs:
        initial_scores = bm25rf.formula(tokenize(query))
        ranked = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True) # Sort by best score
        top_docs = ranked[:top_k]
        relevant_docs = set(doc_id for doc_id, _ in top_docs)
        seen_docs.update(relevant_docs) # To be excluded from at next iteration
        save_labels({doc_id: 1 for doc_id in relevant_docs}, query, label_path) # Mark the relevant documents
        if verbose:
            print(f"[Initialization] Using initial top-{top_k} as pseudo-relevant: {list(relevant_docs)}\n")

    # --- Start iterative PR --- #
    prev_top_doc_ids = set()
    prev_top_docs = []

    for iteration in range(max_iter):
        # Compute RSJ weights from current relevant docs
        weights = bm25rf.prf_weights(relevant_docs, df)
        if not weights:
            if verbose:
                print(f"[Iteration {iteration}] No relevant docs stopping.")
            break

        # Score all documents using probabilistic PR
        orig_terms = tokenize(query)
        term_counter = Counter()

        # Collect term frequencies from relevant_docs
        for term, posting in bm25rf.postings.items():
            for doc_id in relevant_docs:
                if doc_id in posting:
                    term_counter[term] += posting[doc_id]

        # Select top-m expansion terms (excluding originals)
        expansion_terms = [
            t for t, _ in term_counter.most_common(5)
            if t not in orig_terms
        ]

        # Build expanded query
        expanded_query = orig_terms + expansion_terms

        # Score all documents using expanded query
        scores = bm25rf.formula(expanded_query)  # use formula() instead of score_rf if you want standard BM25
        if not scores:
            if verbose:
                print(f"[Iteration {iteration}] No more retrievable documents.")
            break

        # Top-k documents for this iteration
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = ranked_docs[:top_k]
        top_doc_ids = set(doc_id for doc_id, _ in top_docs)

        # Compute Precision@k vs previous top-k
        precision = len(top_doc_ids & prev_top_doc_ids) / top_k if prev_top_doc_ids else 0.0

        # Average score change for top-k
        delta_s = None
        if prev_top_docs:
            prev_scores_dict = dict(prev_top_docs)
            delta_s = np.mean([abs(scores.get(doc_id, 0) - prev_scores_dict.get(doc_id, 0)) 
                               for doc_id in top_doc_ids])

        # Iteration info
        if verbose:
            print(f"[Iteration {iteration}] Precision@{top_k} vs prev: {precision:.4f}")
            if delta_s is not None:
                print(f"[Iteration {iteration}] Avg top-k score change: {delta_s:.6f}")
            print(f"[Iteration {iteration}] Top doc IDs: {[doc_id for doc_id, _ in top_docs[:10]]}\n")

        # Save pseudo-labels
        save_labels({doc_id: 1 for doc_id in top_doc_ids}, query, label_path)

        # Convergence check
        if precision >= precision_threshold:
            if verbose:
                print(f"Converged at iteration {iteration} (Precision@{top_k} ≥ {precision_threshold})\n")
            break

        # Update sets for next iteration
        prev_top_doc_ids = top_doc_ids
        prev_top_docs = top_docs
        relevant_docs.update(top_doc_ids)
        seen_docs.update(top_doc_ids)

    return weights

def compute_BM25PR(df, query, invert_idx, label_path,
                     top_k=10, stability_threshold=0.9, max_iter=10,
                     verbose=True):
    """
    Computes Pseudo-relevance feedback.
    """

    # Initialize BM25PR
    bm25rf = BM25PR(invert_idx, k1=1.5, b=0.75)

    # Pseudo-relevance iterations
    weights = pseudo_relevance_loop(
        query=query,
        bm25rf=bm25rf,
        df=df,
        label_path=label_path,
        top_k=top_k,
        precision_threshold=stability_threshold,
        max_iter=max_iter,
        verbose=verbose
    )

    if not weights:
        if verbose:
            print("[INFO] No RSJ weights computed, skipping final scoring.")
        return pd.DataFrame()

    # Final scoring
    terms = list(weights.keys())
    w_array = np.array([weights[t] for t in terms])

    doc_scores = Counter()
    # Using w = log(p_i/(1-p_i)) + log((1-u_i)/u_i) ~= idf
    for i, term in enumerate(terms):
        postings = bm25rf.postings.get(term, {})
        for doc_id, tf in postings.items():
            dl = bm25rf.doc_len[doc_id]
            tf_norm = bm25rf.Tf(tf, dl)
            doc_scores[doc_id] += w_array[i] * tf_norm

    # Select top-k docs
    top_docs = doc_scores.most_common(top_k)
    top_doc_ids = [doc_id for doc_id, _ in top_docs]
    top_scores = [score for _, score in top_docs]

    # Build DataFrame (For visualisation)
    df_scores = df.iloc[top_doc_ids].copy()
    df_scores["score"] = top_scores
    df_scores = df_scores.sort_values("score", ascending=False)

    # Display results
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print("\n--- Top BM25+PR Documents ---")
    print(df_scores[['id', 'title', 'year', 'score']].to_string(index=False))
    return df_scores


def compute_okapi(df, query, invert_idx, tpk=10):
    query_terms = tokenize(query)
    bm25 = BM25(invert_idx, k1=1.5, b=0.75)
    scores = bm25.formula(query_terms)

    # Build DataFrame (For visualisation)
    df["score"] = df.index.map(scores).fillna(0)
    df = df.sort_values("score", ascending=False)

    # Display results
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print("\n--- Top BM25+PR Documents ---")
    print(df.loc[df["score"] > 0, ["id", "title", "year", "score"]].head(10))
    return df


# Main: insert variables through log
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 / BM25-PR Retrieval")
    parser.add_argument("--bm25rf", action="store_true", help="Enable BM25 pseudo-relevance feedback")
    parser.add_argument("--query", type=str, default="research", help="Search query")
    parser.add_argument("--csv", type=str, default="./openalex10.csv", help="Path to CSV dataset")
    parser.add_argument("--index_path", type=str, default="./index.pkl", help="Path to inverted index")
    parser.add_argument("--label_path", type=str, default="./label.json", help="Path to labels for pseudo-relevance")
    parser.add_argument("--tpk", type=int, default=10, help="How many documents to retrieve")
    parser.add_argument("--max_iter", type=int, default=10, help="Max PR iterations")
    parser.add_argument("--stability", type=float, default=0.9, help="PR convergence threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Load Data
    start = time.time()
    df = pd.read_csv(args.csv).fillna("").reset_index(drop=True)
    df = cleanup(df)
    stop = time.time()
    print(f"[INFO] Data loaded and cleaned in {stop - start:.2f}s")


    # Load or Build Inverted Index
    if not os.path.exists(args.index_path):
        print("[INFO] Building inverted index...")
        invert_idx = inverted_index(df)
        save_index(invert_idx, args.index_path)
    else:
        invert_idx = load_index(args.index_path)
        print(f"[INFO] Inverted index loaded from {args.index_path}")

    # Retrieval
    if args.bm25rf:
        if args.verbose:
            print(f"[INFO] Running BM25 + PR for query: '{args.query}'")
        compute_BM25PR(
            df=df,
            query=args.query,
            invert_idx=invert_idx,
            label_path=args.label_path,
            top_k=args.tpk,
            stability_threshold=args.stability,
            max_iter=args.max_iter,
            verbose=args.verbose
        )
    else:
        if args.verbose:
            print(f"[INFO] Running vanilla BM25 for query: '{args.query}'")
        compute_okapi(df, args.query, invert_idx, args.tpk)