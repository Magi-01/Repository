import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc

def get_metrics(selected_docs, relevant_docs, top_k=None):
    """
    Calculate precision, recall, and F-score using sklearn.

    Args:
        selected_docs (list): List of retrieved document IDs.
        relevant_docs (list): List of relevant document IDs.
        top_k (int, optional): Cutoff for precision@k and recall@k. Defaults to None.

    Returns:
        tuple: (precision, precision@5, precision@10, recall, f_score)
    """
    # Create binary relevance vectors
    all_docs = list(set(selected_docs) | set(relevant_docs))
    y_true = [1 if doc in relevant_docs else 0 for doc in all_docs]
    y_pred = [1 if doc in selected_docs else 0 for doc in all_docs]

    # Precision, recall, F1
    precision_value = precision_score(y_true, y_pred, zero_division=0)
    recall_value = recall_score(y_true, y_pred, zero_division=0)
    f_score_value = f1_score(y_true, y_pred, zero_division=0)

    # Precision@5 and Precision@10
    def precision_at_k(selected, relevant, k):
        if len(selected) == 0:
            return 0.0
        top_k_selected = selected[:k]
        relevant_in_top_k = len([doc for doc in top_k_selected if doc in relevant])
        return relevant_in_top_k / k

    precision_5 = precision_at_k(selected_docs, relevant_docs, 5)
    precision_10 = precision_at_k(selected_docs, relevant_docs, 10)

    return precision_value, precision_5, precision_10, recall_value, f_score_value


def plot_precision_recall_curve(selected_docs, relevant_docs):
    """
    Plot precision-recall curve using sklearn and save as PNG.

    Args:
        selected_docs (list): List of retrieved document IDs.
        relevant_docs (list): List of relevant document IDs.
    """
    if not selected_docs or not relevant_docs:
        return

    all_docs = list(set(selected_docs) | set(relevant_docs))
    y_true = [1 if doc in relevant_docs else 0 for doc in all_docs]
    y_scores = [1 if doc in selected_docs else 0 for doc in all_docs]

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.step(recall, precision, where='post', label=f'PR curve (AUC={pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    image_path = os.path.join("plots", "precision_recall_curve.png")
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path, format='png')
    plt.close()
