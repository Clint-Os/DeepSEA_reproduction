import tensorflow as tf
import json
import numpy as np
from pathlib import Path 
from tokenizer import tokenize_sequence
from keras import models
import argparse 
from keras.models import load_model, model_from_json 
# Removed redundant and incorrect import

MODEL_DIR = Path("models/protein_cnn")
MODEL_PATH = MODEL_DIR / "protein_cnn_model.keras"
LABELS_PATH = MODEL_DIR / "label_to_index.json"
WEIGHTS_PATH = MODEL_DIR / "protein_cnn_weights.h5"
CONFIG_PATH = MODEL_DIR / "model_config.json"

def load_model_and_labels():
    print("Loading model...")
    if CONFIG_PATH.exists() and WEIGHTS_PATH.exists():
        print("Loading model from config and weights...")
        with open(CONFIG_PATH, 'r') as f:
            config_json = f.read()
        #patch for Keras 3/TF 2.17+
        config_json = config_json.replace('"Functional"', '"Model"')

        model = model_from_json(config_json)
        if not model:
            raise ValueError("Failed to load model from configuration.")
        model.load_weights(str(WEIGHTS_PATH))

    elif MODEL_PATH.exists():
        # FALLBACK TO FULL MODEL
        print("Loading model from Keras model file...")
        model = load_model(str(MODEL_PATH))
    
    else:
        raise FileNotFoundError("No model files found.")
    
    with open(LABELS_PATH, 'r') as f:
        label_to_index = json.load(f)
    index_to_label = {v: k for k, v in label_to_index.items()}

    if not model:
        raise ValueError("Model loading failed. Ensure model files are present and valid.")
    return model, index_to_label

def predict_sequence(seq: str, max_len: int = 512):
    model, index_to_label = load_model_and_labels()

    tokenized_seq = tokenize_sequence(seq, max_len)
    tokenized_seq = np.expand_dims(tokenized_seq, axis=0)  # Add batch dimension
    
    #predict
    probs = model.predict(tokenized_seq, verbose = 0)[0]
    pred_index = int(np.argmax(probs, axis=1)[0]) # Get the index of the highest probability
    pred_label = index_to_label[pred_index]

    return pred_label, probs[0] 

def explain_predictions(seq: str, max_len: int = 512, top_k:int=3):
    model, index_to_label = load_model_and_labels()

    tokenized_seq = tokenize_sequence(seq, max_len)
    tokenized_seq = np.expand_dims(tokenized_seq, axis=0)  # Add batch dimension

    # Predict
    probs = model.predict(tokenized_seq, verbose=0)[0]


    # Get top-k predictions
    top_k_indices = np.argsort(probs)[-top_k:][::-1]
    top_k_labels = [index_to_label[idx] for idx in top_k_indices]
    top_k_probs = [float(probs[0][idx]) for idx in top_k_indices]

    explanations = {label: prob for label, prob in zip(top_k_labels, top_k_probs)}
    return explanations

def main():
    parser = argparse.ArgumentParser(description="Run inference on a Protein Sequence Classifier")
    parser.add_argument("--seq", type=str, required=True, help="Protein sequence to classify")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to explain")
    args = parser.parse_args()

    label, probs = predict_sequence(args.seq, args.max_len)
    print(f"Predicted label: {label} with probabilities: {probs}")


    explanations = explain_predictions(args.seq, args.max_len, args.top_k)
    print(f"Top-{args.top_k} predictions:")
    for label, score in explanations:
        print(f"{label}: {score:.4f}")

if __name__ == "__main__":
    main()
