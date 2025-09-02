import os
from pathlib import Path
from Bio import SeqIO
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "merged"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_files = [DATA_DIR / f"CrossVal-training-K{k}.fasta" for k in range(1, 6)]
val_files = [DATA_DIR / f"CrossVal-validation-K{k}.fasta" for k in range(1, 6)]

merged_train = OUTPUT_DIR / "merged_crossval_training.fasta"
merged_val = OUTPUT_DIR / "merged_crossval_validation.fasta"

def merge_fasta(file_list: list[Path], output_file: Path):
    """
    Merges data from multiple fold files into a single file.
    """ 
    records = []
    for f in file_list:
        print(f"Reading {f} ...")
        records.extend(list(SeqIO.parse(f, "fasta")))
    
    print(f"Writing {len(records)} sequences to {output_file} ...")
    SeqIO.write(records, str(output_file), "fasta")
    print("Done.")

def main():
    merge_fasta(train_files, merged_train)
    merge_fasta(val_files, merged_val)
    print("Merging completed. Files saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()