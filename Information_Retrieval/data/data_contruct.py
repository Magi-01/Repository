import boto3
import gzip
import json
import csv
from io import TextIOWrapper
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore import UNSIGNED
from botocore.client import Config
import os

# -----------------------------
# Configuration
# -----------------------------
manifest_file = "manifest.json"
output_csv = "openalex7.csv"
max_bytes = 10 * 1024**3  # 10 GB CSV limit
max_files = 100_000
fields = ["ID", "title", "abstract", "abstract_inverted_index", "year"]
num_threads = 8  # adjust for CPU/network

# -----------------------------
# Setup S3 client
# -----------------------------
s3 = boto3.client(
    "s3",
    region_name="eu-central-1",
    config=Config(signature_version=UNSIGNED)
)

# -----------------------------
# Load manifest and get .gz file keys
# -----------------------------
with open(manifest_file) as f:
    manifest = json.load(f)

files = [entry['url'].replace("s3://openalex/", "")
         for entry in manifest.get("entries", [])
         if entry['url'].endswith(".gz")]

files = files[:max_files]
print(f"Processing {len(files)} files from manifest...")

# -----------------------------
# Helper functions
# -----------------------------
def reconstruct_abstract(inv_index):
    if not inv_index:
        return ""
    max_pos = max(pos for positions in inv_index.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inv_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)

def flatten_work(work):
    inv_index = work.get("abstract_inverted_index")
    return {
        "ID": work.get("id", ""),
        "title": work.get("display_name", ""),
        "abstract": reconstruct_abstract(inv_index),
        "abstract_inverted_index": json.dumps(inv_index or {}),
        "year": work.get("publication_year", "")
    }

def process_file(key):
    results = []
    try:
        obj = s3.get_object(Bucket="openalex", Key=key)
        body = obj['Body']
        with gzip.GzipFile(fileobj=body) as gz:
            for line in TextIOWrapper(gz, encoding='utf-8'):
                work = json.loads(line)
                # Accept all works that are accessible and in English
                if work.get("lang") == "en":
                    results.append(flatten_work(work))
    except Exception as e:
        print(f"Error processing {key}: {e}")
    return results

# -----------------------------
# Parallel processing
# -----------------------------
temp_csvs = []
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = {executor.submit(process_file, key): key for key in files}
    for i, future in enumerate(as_completed(futures), 1):
        key = futures[future]
        data = future.result()
        if data:
            temp_file = f"temp_{i}.csv"
            temp_csvs.append(temp_file)
            with open(temp_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)

        if i % 1000 == 0:
            print(f"Processed {i} files...")

# -----------------------------
# Merge temp CSVs into final CSV
# -----------------------------
total_bytes = 0
with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=fields)
    writer.writeheader()
    for temp_file in temp_csvs:
        with open(temp_file, "r", encoding="utf-8") as f_in:
            next(f_in)  # skip header
            for line in f_in:
                total_bytes += len(line.encode("utf-8"))
                if total_bytes >= max_bytes:
                    print("Reached 10 GB CSV limit. Stopping.")
                    break
                f_out.write(line)
        os.remove(temp_file)

print(f"Finished writing {output_csv}")
