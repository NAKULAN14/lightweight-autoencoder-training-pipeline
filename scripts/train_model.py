import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    balanced_accuracy_score,
    precision_score,
    recall_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =========================================================
# PATHS CONFIGURATION
# =========================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) 

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(PARENT_DIR, "Output") 

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================
print("\nLoading datasets...\n")

X_train              = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_val                = np.load(os.path.join(DATA_DIR, "X_val.npy"))
X_test_benign        = np.load(os.path.join(DATA_DIR, "X_test_benign.npy"))
X_test_attacks       = np.load(os.path.join(DATA_DIR, "X_test_attacks.npy"))
y_test_attack_labels = np.load(os.path.join(DATA_DIR, "y_test_attack_labels.npy"))

print(f"  X_train             : {X_train.shape}")
print(f"  X_val               : {X_val.shape}")
print(f"  X_test_benign      : {X_test_benign.shape}")
print(f"  X_test_attacks     : {X_test_attacks.shape}")
print(f"  y_test_attack_labels: {y_test_attack_labels.shape}")
print("\nDatasets loaded successfully.\n")

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
# BUILD LATEST ADVANCED AUTOENCODER ARCHITECTURE
# =========================================================
print("Building autoencoder model...\n")
input_dim = X_train.shape[1]      

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

autoencoder = Model(inputs=input_layer, outputs=output_layer)

# =========================================================
# COMPILE MODEL WITH HUBER LOSS
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

checkpoint_path = os.path.join(MODELS_DIR, "lightweight_autoencoder.keras")
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    save_best_only=True
)

# =========================================================
# TRAIN MODEL
# =========================================================
print("\nStarting training...\n")
history = autoencoder.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=100,
    batch_size=128,
    callbacks=[early_stop, checkpoint],
    shuffle=True
)
print("\nTraining completed.\n")

# Save as SavedModel format for pure TFLite compilation matching legacy pipelines
savedmodel_path = os.path.join(MODELS_DIR, "autoencoder_savedmodel")
autoencoder.export(savedmodel_path)
print("SavedModel saved for TFLite conversion.\n")

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
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "training_loss.png"))
plt.close()
print("Training loss graph saved.\n")

# =========================================================
# VALIDATION SPLIT FOR THRESHOLD OPTIMIZATION
# =========================================================
X_val_benign = X_val

# Split using explicit index tracking arrays to maintain alignment for Table II
indices = np.arange(len(X_test_attacks))
val_attack_idx, test_attack_idx = train_test_split(
    indices,
    test_size=0.8,
    random_state=42
)

X_val_attack = X_test_attacks[val_attack_idx]
X_test_attacks_final = X_test_attacks[test_attack_idx]
y_test_attack_labels_final = y_test_attack_labels[test_attack_idx]

# =========================================================
# COMPUTE VALIDATION RECONSTRUCTION ERRORS
# =========================================================
val_benign_recon = autoencoder.predict(X_val_benign)
val_benign_mse = np.mean(np.square(X_val_benign - val_benign_recon), axis=1)

val_attack_recon = autoencoder.predict(X_val_attack)
val_attack_mse = np.mean(np.square(X_val_attack - val_attack_recon), axis=1)

y_val = np.concatenate([np.zeros(len(val_benign_mse)), np.ones(len(val_attack_mse))])
val_scores = np.concatenate([val_benign_mse, val_attack_mse])

# Precision-Recall Optimization Curve
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
threshold = thresholds[best_idx]

print(f"Optimized Threshold: {threshold:.6f}")
print(f"Best Validation F1 Score: {f1_scores[best_idx]:.4f}")

threshold_path = os.path.join(MODELS_DIR, "anomaly_threshold.txt")
with open(threshold_path, "w") as f:
    f.write(str(threshold))
print("Threshold saved to models/anomaly_threshold.txt\n")

# =========================================================
# TEST INFERENCE SUITE
# =========================================================
print("Testing on benign traffic...\n")
benign_recon = autoencoder.predict(X_test_benign)
benign_mse = np.mean(np.square(X_test_benign - benign_recon), axis=1)

print("Testing on attack traffic...\n")
attack_recon = autoencoder.predict(X_test_attacks_final)
attack_mse = np.mean(np.square(X_test_attacks_final - attack_recon), axis=1)

y_true = np.concatenate([np.zeros(len(benign_mse)), np.ones(len(attack_mse))])
y_scores = np.concatenate([benign_mse, attack_mse])
y_pred = (y_scores > threshold).astype(int)

# =========================================================
# OVERALL BINARY CLASSIFICATION REPORT & ACCURACY
# =========================================================
print("\n===== OVERALL CLASSIFICATION REPORT =====\n")
report = classification_report(y_true, y_pred, target_names=["Benign", "Attack"], digits=6)
print(report)

global_acc_pct = accuracy_score(y_true, y_pred) * 100
balanced_acc_pct = balanced_accuracy_score(y_true, y_pred) * 100

print(f"Global Model Accuracy   : {global_acc_pct:.2f}%")
print(f"Balanced Model Accuracy : {balanced_acc_pct:.2f}%\n")

eval_report_path = os.path.join(OUTPUTS_DIR, "evaluation_report.txt")
with open(eval_report_path, "w") as f:
    f.write("OVERALL BINARY CLASSIFICATION REPORT\n")
    f.write("=" * 40 + "\n")
    f.write(report)
    f.write(f"\nGlobal Model Accuracy: {global_acc_pct:.2f}%\n")
    f.write(f"Balanced Model Accuracy: {balanced_acc_pct:.2f}%\n")

# =========================================================
# PER-ATTACK-FAMILY F1 SCORES (Table II in IEEE paper)
# =========================================================
ATTACK_LABEL_MAP = {
    0: 'mirai_scan', 1: 'mirai_ack', 2: 'mirai_syn', 3: 'mirai_udp', 4: 'mirai_udpplain',
    5: 'gafgyt_scan', 6: 'gafgyt_junk', 7: 'gafgyt_tcp', 8: 'gafgyt_udp', 9: 'gafgyt_combo',
}

print("\n===== PER-ATTACK-FAMILY F1 SCORES (Table II) =====\n")
print(f"{'Attack Family':<22} {'Samples':>10}  {'F1 Score':>10}")
print("-" * 48)

per_family_results = {}
for label_int, label_name in ATTACK_LABEL_MAP.items():
    mask = (y_test_attack_labels_final == label_int)
    if mask.sum() == 0:
        continue

    family_mse = attack_mse[mask]
    family_pred = (family_mse > threshold).astype(int)
    family_true = np.ones(len(family_pred), dtype=int)

    f1 = f1_score(family_true, family_pred, zero_division=0)
    per_family_results[label_name] = {
        "f1": round(float(f1), 4),
        "samples": int(mask.sum())
    }
    print(f"  {label_name:<20} {mask.sum():>10,}  {f1:>10.4f}")

overall_attack_f1 = f1_score(
    np.ones(len(attack_mse)),
    (attack_mse > threshold).astype(int),
    zero_division=0
)
print("-" * 48)
print(f"  {'OVERALL ATTACK F1':<20} {len(attack_mse):>10,}  {overall_attack_f1:>10.4f}")

family_f1_path = os.path.join(OUTPUTS_DIR, "per_family_f1.txt")
with open(family_f1_path, "w") as f:
    f.write("PER-ATTACK-FAMILY F1 SCORES\n")
    f.write("=" * 40 + "\n")
    f.write(f"{'Family':<22} {'Samples':>10}  {'F1':>8}\n")
    f.write("-" * 44 + "\n")
    for name, vals in per_family_results.items():
        f.write(f"{name:<22} {vals['samples']:>10,}  {vals['f1']:>8.4f}\n")
    f.write("-" * 44 + "\n")
    f.write(f"{'OVERALL ATTACK F1':<22} {len(attack_mse):>10,}  {overall_attack_f1:>8.4f}\n")

print("\nPer-family F1 saved to outputs/per_family_f1.txt\n")

# =========================================================
# RANDOM FOREST BASELINE
# =========================================================
print("===== RANDOM FOREST BASELINE =====\n")
n_benign = len(X_train)
n_attack_rf = min(len(X_test_attacks_final), n_benign)

rng = np.random.default_rng(42)
attack_idx = rng.choice(len(X_test_attacks_final), size=n_attack_rf, replace=False)

X_rf = np.concatenate([X_train[:n_benign], X_test_attacks_final[attack_idx]])
y_rf = np.concatenate([np.zeros(n_benign), np.ones(n_attack_rf)])

shuffle_idx = rng.permutation(len(X_rf))
X_rf, y_rf = X_rf[shuffle_idx], y_rf[shuffle_idx]

X_rf_test = np.concatenate([X_test_benign, X_test_attacks_final])
y_rf_test = np.concatenate([np.zeros(len(X_test_benign)), np.ones(len(X_test_attacks_final))])

print("Training Random Forest (100 trees)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_rf, y_rf)
rf_pred = rf.predict(X_rf_test)

rf_report = classification_report(y_rf_test, rf_pred, target_names=["Benign", "Attack"], digits=6)
print("\nRandom Forest Classification Report:\n")
print(rf_report)

with open(eval_report_path, "a") as f:
    f.write("\n\nRANDOM FOREST BASELINE\n")
    f.write("=" * 40 + "\n")
    f.write(rf_report)
print("Random Forest results appended to outputs/evaluation_report.txt\n")

# =========================================================
# ROC CURVE — COMPARISON
# =========================================================
fpr_ae, tpr_ae, _ = roc_curve(y_true, y_scores)
fpr_rf, tpr_rf, _ = roc_curve(y_rf_test, rf.predict_proba(X_rf_test)[:, 1])
auc_ae = auc(fpr_ae, tpr_ae)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_ae, tpr_ae, label=f"Autoencoder  AUC = {auc_ae:.4f}", linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest AUC = {auc_rf:.4f}", linewidth=2, linestyle='--')
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Autoencoder vs Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "roc_curve.png"))
plt.close()
print("ROC curve saved to outputs/roc_curve.png\n")

# =========================================================
# RECONSTRUCTION ERROR HISTOGRAM (LOG SCALED)
# =========================================================
plt.figure(figsize=(10, 6))
plt.hist(benign_mse, bins=100, alpha=0.6, label='Benign', color='steelblue', log=True)
plt.hist(attack_mse, bins=100, alpha=0.6, label='Attack', color='tomato', log=True)
plt.axvline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold = {threshold:.5f}')
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Frequency (Log Scale)")
plt.title("Reconstruction Error Distribution — Benign vs Attack")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "histogram.png"))
plt.close()
print("Histogram saved to outputs/histogram.png\n")

# =========================================================
# SUMMARY REPORT TERMINAL PRINT OUT
# =========================================================
print("\n" + "=" * 48)
print("  MODEL TRAINING AND EVALUATION COMPLETE")
print("=" * 48)
print(f"\n  Threshold           : {threshold:.6f}")
print(f"  Global Model Acc    : {global_acc_pct:.2f}%")  
print(f"  Balanced Model Acc  : {balanced_acc_pct:.2f}% ")
print(f"  Overall Attack F1   : {overall_attack_f1:.4f}")
print(f"  Autoencoder AUC     : {auc_ae:.4f}")
print(f"  Random Forest AUC   : {auc_rf:.4f}")
print("\n  Files saved:")
print("    models/lightweight_autoencoder.keras")
print("    models/autoencoder_savedmodel/     (for TFLite)")
print("    models/anomaly_threshold.txt       (copy to Raspberry Pi)")
print("    outputs/training_loss.png")
print("    outputs/histogram.png")
print("    outputs/roc_curve.png")
print("    outputs/evaluation_report.txt")
print("    outputs/per_family_f1.txt")
print("\n  Next step: run phase3_tflite_convert.py")
print("=" * 48 + "\n")
