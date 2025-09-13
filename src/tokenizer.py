import tensorflow as tf
from keras import layers
from pathlib import Path
from Bio import SeqIO
from keras.layers import StringLookup

#standard amino acids alphabet
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

VOCAB = AMINO_ACIDS

tokenizer = StringLookup(
    vocabulary=VOCAB,
    mask_token=PAD_TOKEN,
    oov_token=UNK_TOKEN,
    num_oov_indices=1,
)

def tokenize_sequence(seq:str, max_len: int = 512):
    tokens = tokenizer(tf.constant([seq]))
    tokens = tf.squeeze(tokens, axis=0)
    tokens = tokens [:max_len]
    token_shape = tokens.shape[0]
    tokens = tf.pad(tokens, [[0, tf.cast(max_len - token_shape, tf.int32)]], constant_values=0)
    return tokens

def tokenize_fasta(fasta_path: Path, max_len: int = 512):

    sequences, labels = [], []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        label = record.description.split("|")[-1]
        tokens = tokenize_sequence(str(record.seq), max_len = max_len)
        sequences.append(tokens)
        labels.append(label)

    X = tf.stack(sequences)
    return X, labels 

# Add this near the bottom of tokenizer.py

# Reverse lookup: id → token
index_to_token = StringLookup(
    vocabulary=VOCAB,
    mask_token=None,
    invert=True
)

def decode_sequence(tokens):
    """
    Convert a tensor of indices back into an amino acid sequence.
    Removes [PAD] tokens.
    """
    if hasattr(tokens, "numpy"):  # if it's a tf.Tensor
        tokens = tokens.numpy()
    chars = index_to_token(tokens).numpy().astype(str)
    # Drop padding tokens
    chars = [c for c in chars if c not in (PAD_TOKEN,)]
    return "".join(chars)


