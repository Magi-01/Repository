Instructions
python main.py \
  --bm25rf \
  --query "Query:  default "research"" \
  --csv "/path/to/.csv" \
  --index_path "/path/to/.pkl:  default current location" \
  --label_path "/path/to/.json:  default current location" \
  --tpk "int: default 10" \
  --max_iter "int: default 20" \
  --stability "float: default 0.95" \
  --verbose

Order of files is
BM25.py - MAIN:
    Utility.py - where the BM25 and BM25-Pseudo-Relevance classes live
    save_load.py - file for saving and loading inverted index and labels
openalsex_papers5.csv - original file
    openalex10.csv - the file that is worked on
plot.py - for the graphs