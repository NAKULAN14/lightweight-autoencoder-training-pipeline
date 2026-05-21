import numpy as np
import pickle

files = [
    "X_train.npy",
    "X_val.npy",
    "X_test_benign.npy",
    "X_test_attacks.npy",
    "y_test_attack_labels.npy"
]

print("\n========== CHECKING DATA FILES ==========\n")

for file in files:

    data = np.load(f"data/processed/{file}", allow_pickle=True)

    print(f"\nFILE: {file}")
    print("Shape:", data.shape)
    print("Datatype:", data.dtype)

    if data.dtype != object:
        print("NaN count:", np.isnan(data).sum())
        print("Inf count:", np.isinf(data).sum())

        print("Min value:", np.min(data))
        print("Max value:", np.max(data))

print("\n========== CHECKING PICKLE FILE ==========\n")

with open("data/pipeline/kept_features.pkl", "rb") as f:
    features = pickle.load(f)

print("Number of features:", len(features))

print("\nFirst 10 features:\n")
print(features[:10])

print("\n========== DATA CHECK COMPLETED ==========\n")