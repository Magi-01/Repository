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
