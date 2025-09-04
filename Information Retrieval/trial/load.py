
def load_queries(queries_file):
    """
    Load queries from a file and return as a list of tuples.
    Each tuple contains query number and query text.
    """
    with open(queries_file, 'r') as file:
        queries = [line.strip().split('|', 1) for line in file]
    return queries


def load_judgements(file_path):
    """
    Load relevance judgements from a file and return as a dictionary.
    Keys are query numbers, and values are lists of relevant references.
    """
    relevant_refs_by_query = {}

    with open(file_path, 'r') as file:
        current_query_number = None
        current_relevant_refs = []
        relevant_refs_line = False

        for line in file:
            line = line.strip()

            if line.startswith("Query"):
                current_query_number = int(line.split()[1])
                relevant_refs_by_query[current_query_number] = []

            elif "Relevant Refs" in line:
                relevant_refs_line = True

            elif line.strip() and relevant_refs_line:
                current_relevant_refs = line.split()
                for val in current_relevant_refs[:-1]:
                    relevant_refs_by_query[current_query_number].append(val)
                relevant_refs_line = False

    return relevant_refs_by_query