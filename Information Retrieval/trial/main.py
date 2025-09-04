import os
import numpy as np
import preprocces
import indexer
import prob
import metrics

# Constants / parameters
DOCS_FILE = "C:\\Users\\mutua\\Documents\\Repository\\Repository\\Information Retrieval\\trial\\openalex_papers.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

K = 1.2        # BM25 k parameter
B = 0.75       # BM25 b parameter
TOP_N = 10     # Number of top results to display per query

# Queries with relevant documents
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


def display_results_table(results, documents, top_n=10):
    """
    Display top-N search results with document title only.
    """
    print(f"{'Rank':<5} {'Doc ID':<6} {'Title'}")
    print("-"*60)
    doc_map = {str(doc["id"]): doc for doc in documents}
    for rank, (doc_id, score) in enumerate(results[:top_n], start=1):
        title = doc_map.get(str(doc_id), {}).get("title", "N/A")
        print(f"{rank:<5} {doc_id:<6} {title}")


def main():
    # Index documents first
    Inverse = True
    Tokenize = True
    PorterStemmer = True

    print("Indexing documents...")
    index_data = indexer.index(DOCS_FILE, Inverse, Tokenize, PorterStemmer)

    # Load full documents
    documents = preprocces.extract_documents(DOCS_FILE)

    all_metrics = []

    for q_idx, (query_text, relevant_docs) in enumerate(test_queries.items(), start=1):
        print("\n" + "="*80)
        print(f"Query {q_idx}: {query_text}")

        # Preprocess query
        processed_query = preprocces.preprocess_query(query_text, Tokenize, PorterStemmer)

        # BM25 search
        file_path = preprocces.file(Tokenize, PorterStemmer, Inverse)
        search_results = prob.BM25(processed_query, file_path, K, B)

        # Prepare selected docs for metrics
        selected_docs = [str(doc_id) for doc_id, _ in search_results]
        relevant_docs_str = [str(doc) for doc in relevant_docs]
        selected_relevant_docs = [doc for doc in selected_docs if doc in relevant_docs_str]

        # Compute metrics
        precision_val, p5, p10, recall_val, fscore_val = metrics.get_metrics(
            selected_docs, selected_relevant_docs, relevant_docs
        )
        all_metrics.append((precision_val, p5, p10, recall_val, fscore_val))

        # Display top results
        print("\nTop search results:")
        display_results_table(search_results, documents, top_n=TOP_N)

        # Display metrics
        print("\nMetrics:")
        print(f"Precision = {precision_val:.4f}")
        print(f"P@5 = {p5:.4f}")
        print(f"P@10 = {p10:.4f}")
        print(f"Recall = {recall_val:.4f}")
        print(f"F-score = {fscore_val:.4f}")

        # Generate precision-recall plot
        plot_file = os.path.join(PLOTS_DIR, f"precision_recall_query{q_idx}.png")
        metrics.plot_precision_recall_curve(selected_docs, relevant_docs)
        print(f"Precision-recall curve saved to: {plot_file}")

    # Average metrics
    if all_metrics:
        avg_metrics = [sum(m)/len(m) for m in zip(*all_metrics)]
        print("\n" + "="*80)
        print("Average metrics over all queries:")
        print(f"Precision = {avg_metrics[0]:.4f}")
        print(f"P@5 = {avg_metrics[1]:.4f}")
        print(f"P@10 = {avg_metrics[2]:.4f}")
        print(f"Recall = {avg_metrics[3]:.4f}")
        print(f"F-score = {avg_metrics[4]:.4f}")


if __name__ == "__main__":
    main()
