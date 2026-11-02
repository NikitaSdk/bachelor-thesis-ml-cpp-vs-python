import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
import os

print("Phase 1: Training")

# ----------------------------------------------
# 1) Read training data from CSV
# ----------------------------------------------
df = pd.read_csv("clients_mobile_train_data.csv")
print(f"Read {len(df)} customers from training file.\n")

# Extract features matrix (N × 7)
features = df[['Operator',
               'CallMinutes',
               'SMSCount',
               'InternetMB',
               'LastActivityDays',
               'CustomerAge',
               'Prepaid']].to_numpy(dtype=np.float32)

# ----------------------------------------------
# 2) Normalize by columns
# ----------------------------------------------
col_norms = np.linalg.norm(features, axis=0)
col_norms[col_norms == 0] = 1.0
normalized_features = features / col_norms

# ----------------------------------------------
# 3) Train k-means and measure performance
# ----------------------------------------------
k = 3
start_time = time.time()

kmeans = KMeans(n_clusters=k,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=1e-4,
                random_state=42)

kmeans.fit(normalized_features)
training_time = time.time() - start_time

# Calculate metrics
silhouette = silhouette_score(normalized_features, kmeans.labels_)
intra_cluster_dist = kmeans.inertia_ / len(features)  # Average distance to centroid

centroids = kmeans.cluster_centers_.astype(np.float32)

print("Training Metrics:")
print(f"Training time: {training_time:.7f} seconds")
print(f"Silhouette Score: {silhouette:.5f}")
print(f"Intra-cluster Distance: {intra_cluster_dist:.7f}")

# ----------------------------------------------
# 4) Build TensorFlow model
# ----------------------------------------------
class NearestCentroidModel(tf.Module):
    def __init__(self, centroids, col_norms):
        super().__init__()
        self.centroids = tf.constant(centroids)
        self.inv_norms = tf.constant(1.0 / col_norms.astype(np.float32))

    @tf.function(input_signature=[tf.TensorSpec(shape=(7,), dtype=tf.float32)])
    def predict(self, x):
        x_norm = x * self.inv_norms
        diff = self.centroids - x_norm
        sq = tf.square(diff)
        dists = tf.reduce_sum(sq, axis=1)
        idx = tf.argmin(dists, axis=0, output_type=tf.int32)
        return idx

    @tf.function(input_signature=[tf.TensorSpec(shape=(7,), dtype=tf.float32)])
    def __call__(self, x):
        return self.predict(x)

# Instantiate model
model = NearestCentroidModel(centroids=centroids, col_norms=col_norms)

# ----------------------------------------------
# 5) Export to TFLite
# ----------------------------------------------
# Create temporary SavedModel
TEMP_DIR = "tmp_tflite_model"
if os.path.exists(TEMP_DIR):
    import shutil
    shutil.rmtree(TEMP_DIR)

# Save the model with the correct signature
tf.saved_model.save(
    model,
    TEMP_DIR,
    signatures={
        'serving_default': model.predict.get_concrete_function(
            tf.TensorSpec(shape=(7,), dtype=tf.float32)
        )
    }
)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(TEMP_DIR)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

# Save TFLite model
TFLITE_PATH = "kmeans_centroid_model.tflite"
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"\nSaved TFLite model to {TFLITE_PATH}")
