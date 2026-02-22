import os
import pickle
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")

OUT_DICT = os.path.join(BASE_DIR, "tfidf_lite.json")

print("Loading DataFrame...")
with open(DF_PATH, "rb") as f:
    df = pickle.load(f)

print("Loading Sparse Matrix...")
with open(TFIDF_MATRIX_PATH, "rb") as f:
    matrix = pickle.load(f)

print("Extracting Titles & Normalized Vectors...")
titles = df["title"].astype(str).tolist()

# The TF-IDF matrix is (num_movies, vocab_size). 
# We don't want to store 48,000 dense vectors of size 10,000 (too huge).
# We only store the Non-Zero indices and their corresponding weights! 
# This compresses it massively.

row_data = []

# Scipy Sparse CSR matrices have .indptr, .indices, and .data attributes.
for i in range(matrix.shape[0]):
    start = matrix.indptr[i]
    end = matrix.indptr[i+1]
    
    # Store tuples of (vocab_index, weight)
    row_features = [(int(matrix.indices[j]), float(matrix.data[j])) for j in range(start, end)]
    
    # Pre-compute L2 norm for pure python cosine similarity
    norm = float(np.sqrt(sum(w*w for _, w in row_features)))
    
    row_data.append({
        "title": titles[i].strip(),
        "norm": norm if norm > 0 else 1.0,
        "features": row_features
    })

print(f"Compressing {len(row_data)} records to JSON...")
with open(OUT_DICT, "w") as f:
    json.dump(row_data, f)

print(f"Success! Created {OUT_DICT}")
print("You can now safely delete the massive .pkl and .csv files and remove Pandas/Scipy/Sklearn from Vercel!")
