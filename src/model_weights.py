import tensorflow as tf
from keras import models

model_path = "models/protein_cnn/protein_cnn_model.keras"
if not tf.io.gfile.exists(model_path):
	raise FileNotFoundError(f"Model file not found at {model_path}")
model = models.load_model(model_path)
if model is None:
	raise ValueError("Failed to load the model. Please check the model file.")

import json
config_json = model.to_json()
with open("models/protein_cnn/model_config.json", "w") as f:
	f.write(config_json)
	
model.save_weights("models/protein_cnn/protein_cnn_weights.h5")

print("Model config saved to models/protein_cnn/model_config.json")
print("Model weights saved to models/protein_cnn/protein_cnn_weights.h5") 

