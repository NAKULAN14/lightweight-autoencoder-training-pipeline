import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    LeakyReLU
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
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
# STANDARDIZE DATA
# =========================================================

print("Standardizing data...\n")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_test_benign = scaler.transform(X_test_benign)
X_test_attacks = scaler.transform(X_test_attacks)
# =========================================================
# BUILD AUTOENCODER
# =========================================================

print("Building autoencoder model...\n")

input_dim = 115

input_layer = Input(shape=(input_dim,))


# Encoder
x = Dense(48)(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.05)(x)

x = Dense(16)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)

# Smaller bottleneck
bottleneck = Dense(8)(x)

# Decoder
x = Dense(16)(bottleneck)
x = LeakyReLU(alpha=0.1)(x)

x = Dense(48)(x)
x = LeakyReLU(alpha=0.1)(x)

output_layer = Dense(input_dim, activation='linear')(x)
# Create model
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# =========================================================
# COMPILE MODEL
# =========================================================

autoencoder.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss=tf.keras.losses.Huber()
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
    "models/lightweight_autoencoder.keras",
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

    epochs=100,
    batch_size=128,

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
# COMPUTE THRESHOLD
# =========================================================
# =========================================================
# VALIDATION SPLIT FOR THRESHOLD OPTIMIZATION
# =========================================================

from sklearn.model_selection import train_test_split

# Split validation benign data
X_val_benign = X_val

# Create validation attack subset
X_val_attack, X_test_attacks_final = train_test_split(
    X_test_attacks,
    test_size=0.8,
    random_state=42
)

# =========================================================
# COMPUTE VALIDATION RECONSTRUCTION ERRORS
# =========================================================

val_benign_recon = autoencoder.predict(X_val_benign)

val_benign_mse = np.mean(
    np.square(X_val_benign - val_benign_recon),
    axis=1
)

val_attack_recon = autoencoder.predict(X_val_attack)

val_attack_mse = np.mean(
    np.square(X_val_attack - val_attack_recon),
    axis=1
)
# Combine
y_val = np.concatenate([
    np.zeros(len(val_benign_mse)),
    np.ones(len(val_attack_mse))
])

val_scores = np.concatenate([
    val_benign_mse,
    val_attack_mse
])

# Find best threshold
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_scores
)

f1_scores = (
    2 * precision * recall
    / (precision + recall + 1e-8)
)

best_idx = np.argmax(f1_scores)

threshold = thresholds[best_idx]

print(f"Optimized Threshold: {threshold}")
print(f"Best F1 Score: {f1_scores[best_idx]}")

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

attack_recon = autoencoder.predict(X_test_attacks_final)

attack_mse = np.mean(
    np.square(X_test_attacks_final - attack_recon),
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

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(y_true, y_pred)

precision_metric = precision_score(y_true, y_pred)

recall_metric = recall_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)

print("\n========================================")
print("EVALUATION METRICS")
print("========================================\n")

print(f"Accuracy  : {accuracy:.6f}")
print(f"Precision : {precision_metric:.6f}")
print(f"Recall    : {recall_metric:.6f}")
print(f"F1-Score  : {f1:.6f}")

# Full classification report
report = classification_report(
    y_true,
    y_pred,
    digits=6
)

print("\nClassification Report:\n")
print(report)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

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

from sklearn.metrics import average_precision_score

pr_auc = average_precision_score(
    y_true,
    y_scores
)

print(f"PR-AUC: {pr_auc:.4f}")
f1 = f1_score(y_true, y_pred)

print(f"F1-Score: {f1:.4f}")
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