import numpy as np
import pickle
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Load your processed dataset
X = np.load("X_features.npy")   # Your MFCC features
y = np.load("y_labels.npy")     # Encoded labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Reshape for CNN
X_test = X_test.reshape(X_test.shape[0], 40, 130, 1)

# Load model
model = load_model("music_genre_model.h5")

# Predict
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)

# Save metrics
import json
metrics = {
    "accuracy": float(accuracy),
    "f1_score": float(f1)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved to metrics.json")