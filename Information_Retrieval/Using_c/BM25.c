#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/*
doc_term_matrix: D x V row-major document-term frequency matrix
query_term_matrix: Q x V row-major query-term TF matrix (usually 0/1)
doc_lengths: D array of document lengths
idf: V array of precomputed IDF
scores: Q x D row-major output array
D: number of documents
V: vocabulary size
Q: number of queries
k1, b, avgdl: BM25 hyperparameters
*/

void bm25_batch_multiquery(const int *doc_term_matrix, const int *query_term_matrix,
                           const double *idf, const double *doc_lengths,
                           int D, int V, int Q, double avgdl, double k1, double b,
                           double *scores)
{
    // Parallelize over queries
    #pragma omp parallel for
    for (int q = 0; q < Q; q++) {
        for (int d = 0; d < D; d++) {
            double score = 0.0;
            double len_d = doc_lengths[d];

            for (int t = 0; t < V; t++) {
                int f = doc_term_matrix[d * V + t];     // term frequency in doc
                int qf = query_term_matrix[q * V + t];  // term frequency in query
                if (f > 0 && qf > 0) {
                    double tf = f / (f + k1 * (1.0 - b + b * len_d / avgdl));
                    score += tf * idf[t] * qf;  // multiply by query TF if needed
                }
            }
            scores[q * D + d] = score;
        }
    }
}

void compute_idf(const int *doc_term_matrix, int D, int V, double *idf)
{
    for (int t = 0; t < V; t++) {
        int n_qt = 0;
        for (int d = 0; d < D; d++) {
            if (doc_term_matrix[d * V + t] > 0)
                n_qt++;
        }
        idf[t] = log((D - n_qt + 0.5)/(n_qt + 0.5));
    }
}
