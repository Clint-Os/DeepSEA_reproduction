import random
from pathlib import Path
from Bio import SeqIO
from sklearn.model_selection import train_test_split 

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'Data'/'merged'
INPUT_FILE = DATA_DIR / 'merged_crossval_training.fasta'

OUTPUT_DIR = DATA_DIR / 'train_test_split'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / 'train.fasta'
TEST_FILE = OUTPUT_DIR / 'test.fasta'

TEST_SIZE = 0.30
RANDOM_SEED = 42

def split_fasta(input_file,
                train_file,
                test_file,
                test_size = 0.3,
                random_seed = 42):
    print(f"Reading sequences from {input_file}...")
    sequences = list(SeqIO.parse(input_file, "fasta"))
    print(f"Total sequences read: {len(sequences)}")

    train_records, test_records = train_test_split(
        sequences,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True
    )
    print(f"Training set size: {len(train_records)}| Test size: {len(test_records)}")

    SeqIO.write(train_records, str(train_file), 'fasta')
    SeqIO.write(test_records, str(test_file), 'fasta')

if __name__ == '__main__':
    split_fasta(INPUT_FILE, TRAIN_FILE, TEST_FILE, TEST_SIZE, RANDOM_SEED)
    print(f"Train and test sets saved to {OUTPUT_DIR}")
