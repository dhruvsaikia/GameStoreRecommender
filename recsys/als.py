# recsys/als.py
from typing import Dict, Tuple, Union, Sequence
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares

def build_id_maps(df) -> Tuple[Dict[str, int], Dict[str, int], np.ndarray, np.ndarray]:
    users = df["user"].unique()
    items = df["game"].unique()
    user2id = {u: i for i, u in enumerate(users)}
    item2id = {g: i for i, g in enumerate(items)}
    return user2id, item2id, users, items

def make_sparse(df, user2id, item2id) -> csr_matrix:
    rows = df["user"].map(user2id).values
    cols = df["game"].map(item2id).values
    vals = df["strength"].astype("float32").values
    coo = coo_matrix((vals, (rows, cols)), shape=(len(user2id), len(item2id)))
    return coo.tocsr()  # user × item

def train_als(user_item_csr: csr_matrix, factors: int = 32, reg: float = 0.05, iters: int = 10):
    """
    Train with the matrix as USER × ITEM (no transpose) — matches your implicit build.
    """
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iters
    )
    model.fit(user_item_csr)

    # Sanity: item_factors must match number of items (num columns)
    n_items = user_item_csr.shape[1]
    assert getattr(model, "item_factors").shape[0] == n_items, (
        f"Model items ({model.item_factors.shape[0]}) != matrix items ({n_items})"
    )
    return model

def _to_scalar_int(x: Union[int, np.ndarray, Sequence]) -> int:
    if np.isscalar(x):
        return int(x)
    arr = np.asarray(x).ravel()
    if arr.size == 0:
        raise ValueError("Empty user id provided.")
    return int(arr[0])

def recommend_topn(model, user_idx, user_item_csr: csr_matrix, N: int = 10):
    """
    IMPORTANT for your implicit build:
    - Pass a SINGLE-ROW CSR (that user's row), not the full matrix.
    - Set recalculate_user=True.
    """
    uid = _to_scalar_int(user_idx)

    # Ensure CSR and slice the single user's row
    if not isinstance(user_item_csr, csr_matrix):
        user_item_csr = user_item_csr.tocsr()
    user_row = user_item_csr.getrow(uid).tocsr()

    # Recommend for that user using their single-row matrix
    ids, scores = model.recommend(uid, user_row, N=N, recalculate_user=True)

    # Sanity: ids must be valid column indices
    n_items = user_item_csr.shape[1]
    if len(ids) and (int(np.max(ids)) >= n_items):
        raise RuntimeError(
            f"Recommend returned id >= n_items (max {int(np.max(ids))} vs {n_items}); "
            "training and recommend index spaces mismatch."
        )
    return ids, scores, uid
