from pathlib import Path
from Bio import SeqIO
from sklearn.model_selection import KFold

import os 

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Data'/'merged'/'train_test_split'
INPUT_FILE = DATA_DIR/'train.fasta'

OUTPUT_DIR = BASE_DIR / 'Data'/'merged'/'cv_folds'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
SEED = 42

def get_labels_from_sequences(sequences):
    labels = []
    for record in sequences:
        header = record.description
        
        try:
            label = header.split('|')[-1] #splits the last field after the last '|'
        except:
            label = 'Unknown'
        labels.append(label)
    return labels

def make_cv_folds (input_file, n_splits=5, seed=42):
    print(f"Reading sequences from {input_file}...")
    sequences = list(SeqIO.parse(input_file, "fasta"))
    print(f"Total sequences read: {len(sequences)}")

    labels = get_labels_from_sequences(sequences)

    skf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for i, (train_idx, val_idx) in enumerate(skf.split(sequences, labels), start=1):
        fold_dir = OUTPUT_DIR / f'fold_{i}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_records = [sequences[j] for j in train_idx]
        val_records = [sequences[j] for j in val_idx]

        train_out = fold_dir / 'train.fasta'
        val_out = fold_dir / 'val.fasta'

        SeqIO.write(train_records, str(train_out), "fasta")
        SeqIO.write(val_records, str(val_out), "fasta")

        print(f"Fold {i}: Train={len(train_records)}, Val={len(val_records)}")

if __name__ == "__main__":
    make_cv_folds(INPUT_FILE, n_splits=N_SPLITS, seed=SEED)
print(f"5 CV folds created successfully in {OUTPUT_DIR}")