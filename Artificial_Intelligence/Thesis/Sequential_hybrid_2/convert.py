# convert_column_to_indexed_list.py
from pathlib import Path

BASE_DIR = Path(__file__).parent

input_file = BASE_DIR / "numbers.txt"       # your original file
output_file = BASE_DIR / "indexed_numbers.txt"

with open(input_file, "r") as f:
    lines = f.readlines()

with open(output_file, "w") as f:
    for idx, line in enumerate(lines, start=1):
        num = line.strip()
        if num:  # skip empty lines
            f.write(f'"{idx}": [{num}],\n')

print(f"Saved {len(lines)} entries to {output_file}")