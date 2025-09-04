import os
import nltk
from collections import defaultdict

# Ensure required NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


def tokenization(text):
    """
    Tokenize the given text.

    Args:
        text (str): Input text to tokenize.

    Returns:
        list: List of tokens.
    """
    ExpReg = nltk.RegexpTokenizer(
        r'(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.,]\d+)?%?|\w+(?:[\-/]\w+)*'
    )
    return ExpReg.tokenize(text)


def stop_words(tokens):
    """
    Remove stop words from the list of tokens.

    Args:
        tokens (list): List of tokens.

    Returns:
        list: List of tokens without stop words.
    """
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in tokens if word.lower() not in nltk_stopwords]


def normalization_porter(tokens):
    """
    Perform stemming using the Porter algorithm.

    Args:
        tokens (list): List of tokens.

    Returns:
        list: List of normalized tokens.
    """
    Porter = nltk.PorterStemmer()
    return [Porter.stem(word) for word in tokens]


def normalization_lancaster(tokens):
    """
    Perform stemming using the Lancaster algorithm.

    Args:
        tokens (list): List of tokens.

    Returns:
        list: List of normalized tokens.
    """
    Lancaster = nltk.LancasterStemmer()
    return [Lancaster.stem(word) for word in tokens]


def extract_documents(file_path):
    """
    Extract documents from a CSV file.

    Args:
        file_path (str): Path to CSV file.

    Returns:
        list: List of documents as dictionaries.
    """
    import csv
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = {
                "id": row["id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "year": int(row["year"]),
                "concepts": row["concepts"],
                "abstract_text": row["abstract_text"]
            }
            documents.append(doc)
    return documents


def write_dict_to_file(my_dict, filename):
    """
    Write dictionary content to a file.

    Args:
        my_dict (dict): Dictionary to write to the file.
        filename (str): Name of the output file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        for key, values in my_dict.items():
            for files_list, freq, weight in values:
                file.write(f"{key} {files_list} {freq} {weight:.5f}\n")


def preprocess_query(query, Tokenize, PorterStemmer):
    """
    Preprocess the query text.

    Args:
        query (str): Query text.
        Tokenize (bool): Flag indicating whether to tokenize the query.
        PorterStemmer (bool): Flag indicating whether to use Porter stemming.

    Returns:
        list: Preprocessed query tokens.
    """
    if Tokenize:
        q = tokenization(query)
    else:
        q = query.split()

    q = stop_words(q)

    if PorterStemmer:
        return normalization_porter(q)
    else:
        return normalization_lancaster(q)


def file(Tokenize, PorterStemmer, Inverse=None):
    """
    Generate file path based on Tokenize, PorterStemmer, and Inverse flags.

    Args:
        Tokenize (bool): Tokenization flag.
        PorterStemmer (bool): Porter stemmer flag.
        Inverse (bool, optional): Inverse descriptor flag. Defaults to None.

    Returns:
        str: Full path to file.
    """
    base_dir = os.path.join("Inverses & Descriptors")
    os.makedirs(base_dir, exist_ok=True)

    if Inverse or Inverse is None:
        if Tokenize:
            return os.path.join(base_dir, "InverseTokenPorter.txt" if PorterStemmer else "InverseTokenLancaster.txt")
        else:
            return os.path.join(base_dir, "InverseSplitPorter.txt" if PorterStemmer else "InverseSplitLancaster.txt")
    else:
        if Tokenize:
            return os.path.join(base_dir, "DescripteurTokenPorter.txt" if PorterStemmer else "DescripteurTokenLancaster.txt")
        else:
            return os.path.join(base_dir, "DescripteurSplitPorter.txt" if PorterStemmer else "DescripteurSplitLancaster.txt")
