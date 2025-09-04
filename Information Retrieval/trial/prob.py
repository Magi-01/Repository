import numpy as np
import preprocces
import os

def n_docs_terms(file_path, query):
    """
    Count the number of documents containing each term in the query.
    
    Args:
        file_path (str): Path to the file containing term-document frequencies.
        query (list): List of query terms.

    Returns:
        dict: Dictionary containing term-document counts for query terms.
    """
    documents_containing_terms = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            current_term = parts[0]

            if current_term in query:
                documents_containing_terms[current_term] = documents_containing_terms.get(current_term, 0) + 1

            # Optimization: stop if all query terms have been counted
            if len(documents_containing_terms) == len(query):
                break

    return documents_containing_terms


def BM25(query, file_path, K, B):
    """
    Calculate BM25 scores for documents based on the query.

    Args:
        query (list): List of query terms.
        file_path (str): Path to the file containing term-document frequencies.
        K (float): BM25 constant K.
        B (float): BM25 constant B.

    Returns:
        list: List of tuples containing (document_id, score) sorted by score descending.
    """
    dl = {}          # document lengths
    result = {}      # BM25 scores
    total_terms = 0  # total terms in collection

    # First pass: compute document lengths
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            _, doc_id, freq, _ = parts
            freq = int(freq)
            dl[doc_id] = dl.get(doc_id, 0) + freq
            total_terms += freq

    N = len(dl)
    avdl = total_terms / N if N > 0 else 1
    ni = n_docs_terms(file_path, query)  # documents containing each query term

    # Second pass: compute BM25 scores
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            term, doc_id, freq, _ = parts
            freq = int(freq)

            if term not in query:
                continue

            score = ((freq / (freq + K * ((1 - B) + B * (dl[doc_id] / avdl)))) *
                     np.log10((N - ni[term] + 0.5) / (ni[term] + 0.5)))

            result[doc_id] = result.get(doc_id, 0) + score

    # Sort results by score descending
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    return result


def probabilistic_model(query, Tokenize, PorterStemmer, K, B):
    """
    Perform probabilistic retrieval model using BM25.

    Args:
        query (str): Query text.
        Tokenize (bool): Whether to tokenize the query.
        PorterStemmer (bool): Whether to apply Porter stemming.
        K (float): BM25 K parameter.
        B (float): BM25 B parameter.

    Returns:
        list: List of tuples (document_id, score) sorted by score descending.
    """
    # Preprocess query
    query_terms = preprocces.preprocess_query(query, Tokenize, PorterStemmer)

    # Get index file
    file_path = preprocces.file(Tokenize, PorterStemmer)

    # Compute BM25 scores
    results = BM25(query_terms, file_path, K, B)
    return results
