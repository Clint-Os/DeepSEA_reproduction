import tensorflow as tf 
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import models
from tokenizer import tokenize_fasta
import json as json_module
from build_model import protein_cnn_model 
import os 


#config
DRIVE_FASTA = Path("/content/drive/MyDrive/DeepSEA-project/Data/merged/merged_crossval_training.fasta")
LOCAL_FASTA = Path("Data/merged/merged_crossval_training.fasta")

if DRIVE_FASTA.exists():
    FASTA_PATH = DRIVE_FASTA
    print(f"using GOOGLE DRIVE fasta file: {FASTA_PATH}")
elif LOCAL_FASTA.exists():
    FASTA_PATH = LOCAL_FASTA
else:
    raise FileNotFoundError("No FASTA file found in either location.")

MAX_LEN = 512
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 10
SAVE_DIR = Path("/content/drive/MyDrive/DeepSEA-project/models/protein_cnn")
MODEL_PATH = SAVE_DIR / "protein_cnn_model.keras"
LABELS_PATH = SAVE_DIR / "label_to_index.json"

def main():
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        print(f"Model already exists at {MODEL_PATH}, skipping training..")
        model = models.load_model(MODEL_PATH, compile=False)
        
        with open(LABELS_PATH) as f:
            label_to_index = json_module.load(f)
        print("Model and label mapping loaded...")
        return model, label_to_index
    #load data
    print("Loading and tokenizing data...")
    X, labels = tokenize_fasta(FASTA_PATH, max_len=MAX_LEN)

    #convert labels to integers and then to categorical
    label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    y = [label_to_index[label] for label in labels]
    y = to_categorical(y, num_classes=len(label_to_index))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building protein CNN model...")
    model = protein_cnn_model(
        max_len=MAX_LEN,
        embedding_dim=128,
        num_classes=len(label_to_index),
        conv_filters=[64, 128, 256],
        kernel_size=3,
        dropout_rate=0.3,
    )
    
    model.compile(    
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
    ) 

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose='auto', # Set to 'auto' for progress bar
    )
    print("Training complete.")

    #--- Save our model---
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(SAVE_DIR / "protein_cnn_model.keras", save_format="keras")
    print(f"Model saved to {SAVE_DIR / 'protein_cnn_model.keras'}")

    import json
    with open(SAVE_DIR / "label_to_index.json", "w") as f:
        json.dump(label_to_index, f)
    print(f"Label mapping saved to {SAVE_DIR / 'label_to_index.json'}")

if __name__ == "__main__":
    main()