import nltk
import numpy as np
from collections import defaultdict
import preprocces
import os


def index(file_path, Inverse=False, Tokenize=False, PorterStemmer=False):
    """
    Index the documents in the given file.

    Args:
        file_path (str): Path to the CSV file containing documents.
        Inverse (bool): Flag indicating whether to build inverse index (term -> documents).
        Tokenize (bool): Flag indicating whether to tokenize text.
        PorterStemmer (bool): Flag indicating whether to use Porter stemming.

    Returns:
        dict: Indexed document data.
    """
    word_file_count = defaultdict(set)
    unique_document_numbers = set()
    temp_dict = {}      # Initial mapping: either term->doc or doc->term
    final_dict = {}     # Weighted output

    documents = preprocces.extract_documents(file_path)

    for document in documents:
        file_id = document["id"]
        text = document["abstract_text"]

        # Split by sections if multiple headers exist
        if "\n\n" in text:  # Assuming double line breaks separate sections
            sections = text.split("\n\n")
            text = " ".join(sections)

        unique_document_numbers.add(file_id)

        # Tokenization
        tokens = preprocces.tokenization(text) if Tokenize else text.split()

        # Stop-word removal
        tokens = preprocces.stop_words(tokens)

        # Normalization / stemming
        tokens = preprocces.normalization_porter(tokens) if PorterStemmer else preprocces.normalization_lancaster(tokens)

        # Compute frequencies
        words_frequency = nltk.FreqDist(tokens)
        max_freq = max(words_frequency.values()) if words_frequency else 1

        # Build initial mapping
        for word, freq in words_frequency.items():
            if Inverse:
                temp_dict.setdefault(word, []).append((file_id, freq, max_freq))
            else:
                temp_dict.setdefault(file_id, []).append((word, freq, max_freq))
                word_file_count[word].add(file_id)

    # Compute weighted values (TF-IDF like)
    for key1, values in temp_dict.items():
        for key2, freq, max_freq in values:
            if Inverse:
                weight = (freq / max_freq) * np.log10((len(unique_document_numbers) / len(temp_dict[key1])) + 1)
            else:
                weight = (freq / max_freq) * np.log10((len(unique_document_numbers) / len(word_file_count[key2])) + 1)
            final_dict.setdefault(key1, []).append((key2, freq, weight))

    # Save to file
    output_file = preprocces.file(Tokenize, PorterStemmer, Inverse)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    preprocces.write_dict_to_file(final_dict, output_file)

    return final_dict
