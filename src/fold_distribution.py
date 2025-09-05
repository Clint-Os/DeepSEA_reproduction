from pathlib import Path
from collections import Counter
from Bio import SeqIO

BASE_DIR = Path(__file__).resolve().parent.parent
CV_DIR = BASE_DIR / "Data" / "merged" / "cv_folds"

def count_labels(fasta_file):
    labels = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.description
        label = header.split("|")[-1]
        labels.append(label)
    return Counter(labels)

def inspect_folds(cv_dir):
    for fold_dir in sorted(cv_dir.glob("fold*")):
        print(f"\n=== {fold_dir.name} ===")
        for split in ["train.fasta", "val.fasta"]:
            file_path = fold_dir / split
            label_counts = count_labels(file_path)
            total = sum(label_counts.values())
            print(f"{split}: {total} sequences")
            for label, count in label_counts.items():
                print(f"  {label}: {count}")
                
if __name__ == "__main__":
    inspect_folds(CV_DIR)
