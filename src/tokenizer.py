import tensorflow as tf
from keras import layers
from pathlib import Path
from Bio import SeqIO
from keras.layers import StringLookup
import numpy as np 

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

def tokenize_sequence(seq:str, max_len: int = 512) -> tf.Tensor:

    char_list = list(seq)
    if len(char_list) == 0:
        tokens = tf.constant([], dtype=tf.int32)
    else:
        char_tensor = tf.constant(char_list, dtype=tf.string)
        tokens = tokenizer(char_tensor)
        tokens = tf.cast(tokens, tf.int32) 

    tokens = tf.strided_slice(tokens, [0], [max_len]) # Truncate if necessary
    pad_len = max_len - tf.gather(tf.shape(tokens), 0)
    tokens = tf.pad(tokens, [[0, pad_len]], constant_values=0)
    
    tokens.set_shape([max_len]) 
    return tokens

def tokenize_fasta(fasta_path: Path, max_len: int = 512):

    sequences, labels = [], []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        label = record.description.split("|")[-1]
        tokens = tokenize_sequence(str(record.seq), max_len = max_len)
        sequences.append(tokens)
        labels.append(label)

    X = np.stack([seq.numpy() for seq in sequences])
    y = np.array(labels)
    return X, labels 

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


