from pathlib import Path
from Bio import SeqIO
import os 

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Data'/'merged'/'train_test_split'
INPUT_FILE = DATA_DIR/'train.fasta'

def inspect_header(file_path, n=5):
    print(f'Inspecting first {n} headers from {file_path}...\n')
    for i, record in enumerate(SeqIO.parse(file_path, 'fasta')):
        header = record.description
        label = header.split("|")[-1] if '|' in header else 'Unknown'
        print(f"Header {i+1}: {header}")
        print(f"Extracted Label: {label}\n")
        if i + 1 >= n:
            break

if __name__ == "__main__":
    inspect_header(INPUT_FILE, n=5)
    print(f"Header inspection completed for {INPUT_FILE}")
