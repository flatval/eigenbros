import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from implicit.als import AlternatingLeastSquares
import scipy.sparse as sp
from tqdm import tqdm

class MissingOperationsFiller:
    """Predict missing work operation codes for rooms with improved ALS and k-NN"""

    def __init__(self):
        self.knn_model = None
        self.als_model = None
        self.operation_codes = []
        self.operation_to_idx = {}
        self.idx_to_operation = {}
        self.binary_matrix = None
        self.entity_ids = []

    def fit(self, df_complete: pd.DataFrame, k_neighbors=20, als_factors=100, als_iterations=50):
        df_complete['entity_id'] = df_complete['project_id'].astype(str) + '_' + df_complete['room'].astype(str)
        self.operation_codes = sorted(df_complete['work_operation_cluster_code'].dropna().astype(int).unique())
        self.operation_to_idx = {op: i for i, op in enumerate(self.operation_codes)}
        self.idx_to_operation = {i: op for op, i in self.operation_to_idx.items()}

        self.entity_ids = sorted(df_complete['entity_id'].unique())
        entity_to_idx = {e: i for i, e in enumerate(self.entity_ids)}

        n_entities = len(self.entity_ids)
        n_operations = len(self.operation_codes)
        self.binary_matrix = np.zeros((n_entities, n_operations), dtype=np.int8)

        # Fyll binær matrise og vekt basert på hyppighet
        for _, row in df_complete.iterrows():
            entity_idx = entity_to_idx[row['entity_id']]
            op_code = int(row['work_operation_cluster_code'])
            if op_code in self.operation_to_idx:
                self.binary_matrix[entity_idx, self.operation_to_idx[op_code]] += 1

        # Train k-NN
        self.knn_model = NearestNeighbors(
            n_neighbors=min(k_neighbors, n_entities),
            metric='cosine',  # bedre for vektet input
            algorithm='brute'
        )
        self.knn_model.fit(self.binary_matrix)

        # Train ALS
        sparse_matrix = sp.csr_matrix(self.binary_matrix.T, dtype=np.float32)
        self.als_model = AlternatingLeastSquares(
            factors=als_factors,
            regularization=0.1,
            iterations=als_iterations,
            random_state=42
        )
        weighted_matrix = (sparse_matrix * 40).astype('double')
        self.als_model.fit(weighted_matrix, show_progress=False)

    def predict_missing(self, known_operations, top_k=5, weight_knn=0.6, weight_als=0.4):
        input_vector = np.zeros(len(self.operation_codes), dtype=np.int8)
        valid_known = [op for op in known_operations if op in self.operation_to_idx]
        for op in valid_known:
            input_vector[self.operation_to_idx[op]] = 1
        if not valid_known:
            counts = self.binary_matrix.sum(axis=0)
            top_indices = np.argsort(counts)[-top_k:][::-1]
            return [self.idx_to_operation[i] for i in top_indices]

        # k-NN scores
        distances, indices = self.knn_model.kneighbors(input_vector.reshape(1, -1), return_distance=True)
        op_scores = np.zeros(len(self.operation_codes))
        for dist, idx in zip(distances[0], indices[0]):
            similarity = np.exp(-dist)  # mer differensiering
            op_scores += self.binary_matrix[idx] * similarity
        if op_scores.max() > 0:
            op_scores /= op_scores.max()

        # ALS scores
        als_scores = np.zeros(len(self.operation_codes))
        for op in valid_known:
            idx = self.operation_to_idx[op]
            als_scores += np.dot(self.als_model.user_factors, self.als_model.user_factors[idx])
        if als_scores.max() > 0:
            als_scores /= als_scores.max()

        # Combine
        combined_scores = weight_knn * op_scores + weight_als * als_scores
        for op in valid_known:
            combined_scores[self.operation_to_idx[op]] = 0

        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        return [self.idx_to_operation[i] for i in top_indices if combined_scores[i] > 0]


def generate_submission(csv_path, output_path, top_k=3, max_op_code=387):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df_complete = df[~df['work_operation_cluster_code'].isna()].copy()
    df['entity_id'] = df['project_id'].astype(str) + '_' + df['room'].astype(str)

    filler = MissingOperationsFiller()
    filler.fit(df_complete)

    predictions_list = []
    all_rooms = df['entity_id'].unique()
    for row_idx, entity_id in enumerate(tqdm(all_rooms, desc="Processing rooms")):
        subset = df[df['entity_id'] == entity_id]

        known_ops = df_complete[
            (df_complete['project_id'] == subset['project_id'].iloc[0]) &
            (df_complete['room'] == subset['room'].iloc[0])
        ]['work_operation_cluster_code'].dropna().astype(int).tolist()

        # Dynamisk top_k: flere kjente => predikter flere
        dyn_top_k = max(3, len(known_ops)//2)
        preds = filler.predict_missing(known_ops, top_k=dyn_top_k)

        row = [row_idx] + [0] * (max_op_code + 1)
        for op in preds:
            if 0 <= op <= max_op_code:
                row[op + 1] = 1
        predictions_list.append(row)

    columns = ['id'] + list(range(0, max_op_code + 1))
    df_submission = pd.DataFrame(predictions_list, columns=columns)
    df_submission.to_csv(output_path, index=False)
    print(f"✓ Submission CSV created: {output_path}")
    return df_submission


if __name__ == "__main__":
    CSV_PATH = "test.csv"
    OUTPUT_PATH = "submission.csv"
    TOP_K = 3
    generate_submission(CSV_PATH, OUTPUT_PATH, top_k=TOP_K)
