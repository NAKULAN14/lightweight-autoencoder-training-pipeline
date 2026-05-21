import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc
)

# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading datasets...\n")

X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")

X_test_benign = np.load("data/processed/X_test_benign.npy")
X_test_attacks = np.load("data/processed/X_test_attacks.npy")

y_test_attack_labels = np.load(
    "data/processed/y_test_attack_labels.npy"
)

print("Datasets loaded successfully.\n")

# =========================================================
# BUILD AUTOENCODER
# =========================================================

print("Building autoencoder model...\n")

input_dim = 115

input_layer = Input(shape=(input_dim,))

# Encoder
x = Dense(64, activation='relu')(input_layer)
x = Dense(32, activation='relu')(x)
bottleneck = Dense(16, activation='relu')(x)

# Decoder
x = Dense(32, activation='relu')(bottleneck)
x = Dense(64, activation='relu')(x)

output_layer = Dense(input_dim, activation='sigmoid')(x)

# Create model
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# =========================================================
# COMPILE MODEL
# =========================================================

autoencoder.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)

autoencoder.summary()

# =========================================================
# CALLBACKS
# =========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/lightweight_autoencoder.h5",
    monitor='val_loss',
    save_best_only=True
)

# =========================================================
# TRAIN MODEL
# =========================================================

print("\nStarting training...\n")

history = autoencoder.fit(

    X_train,
    X_train,

    validation_data=(X_val, X_val),

    epochs=50,
    batch_size=256,

    callbacks=[early_stop, checkpoint],

    shuffle=True
)

print("\nTraining completed.\n")

# =========================================================
# SAVE TRAINING LOSS PLOT
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig("outputs/training_loss.png")

print("Training loss graph saved.\n")

# =========================================================
# COMPUTE VALIDATION RECONSTRUCTION ERROR
# =========================================================

print("Computing reconstruction errors...\n")

val_reconstructions = autoencoder.predict(X_val)

val_mse = np.mean(
    np.square(X_val - val_reconstructions),
    axis=1
)

# =========================================================
# COMPUTE THRESHOLD
# =========================================================

threshold = np.mean(val_mse) + 3 * np.std(val_mse)

print(f"Anomaly Threshold: {threshold}")

with open("models/anomaly_threshold.txt", "w") as f:
    f.write(str(threshold))

print("Threshold saved.\n")

# =========================================================
# TEST ON BENIGN DATA
# =========================================================

print("Testing on benign traffic...\n")

benign_recon = autoencoder.predict(X_test_benign)

benign_mse = np.mean(
    np.square(X_test_benign - benign_recon),
    axis=1
)

# =========================================================
# TEST ON ATTACK DATA
# =========================================================

print("Testing on attack traffic...\n")

attack_recon = autoencoder.predict(X_test_attacks)

attack_mse = np.mean(
    np.square(X_test_attacks - attack_recon),
    axis=1
)

# =========================================================
# CREATE LABELS
# =========================================================

y_true = np.concatenate([
    np.zeros(len(benign_mse)),
    np.ones(len(attack_mse))
])

y_scores = np.concatenate([
    benign_mse,
    attack_mse
])

# =========================================================
# PREDICTIONS USING THRESHOLD
# =========================================================

y_pred = (y_scores > threshold).astype(int)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(y_true, y_pred)

print("\nClassification Report:\n")
print(report)

with open("outputs/evaluation_report.txt", "w") as f:
    f.write(report)

# =========================================================
# ROC CURVE
# =========================================================

fpr, tpr, _ = roc_curve(y_true, y_scores)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("outputs/roc_curve.png")

print("ROC curve saved.\n")

# =========================================================
# HISTOGRAM
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    benign_mse,
    bins=100,
    alpha=0.6,
    label='Benign'
)

plt.hist(
    attack_mse,
    bins=100,
    alpha=0.6,
    label='Attack'
)

plt.axvline(
    threshold,
    color='red',
    linestyle='--',
    label='Threshold'
)

plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")

plt.title("Benign vs Attack Reconstruction Error")

plt.legend()

plt.savefig("outputs/histogram.png")

print("Histogram saved.\n")

# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n========================================")
print("MODEL TRAINING AND EVALUATION COMPLETE")
print("========================================\n")