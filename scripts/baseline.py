# scripts/baseline.py
import os
import argparse
import numpy as np

# keep BLAS quiet/predictable (optional)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from recsys.data import load_steam200k
from recsys.als import build_id_maps, make_sparse, train_als, recommend_topn

def main():
    # --- CLI args ---
    parser = argparse.ArgumentParser(description="Steam-200k ALS recommender")
    parser.add_argument("--user", type=str, default=None, help="Steam user id as it appears in the CSV (e.g., 151603712)")
    parser.add_argument("--k", type=int, default=10, help="Number of recommendations to return")
    args = parser.parse_args()

    # 1) Load data -> tidy [user, game, strength]
    df = load_steam200k()

    # Precompute popularity for fallback (game names, most 'strength' first)
    popularity = (
        df.groupby("game")["strength"].sum().sort_values(ascending=False).index.tolist()
    )

    # 2) Build id maps + user-item CSR
    u2i, i2i, users, items = build_id_maps(df)
    mat = make_sparse(df, u2i, i2i)
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        raise RuntimeError("Matrix is empty; check that data/steam-200k.csv exists and loaded.")

    # Resolve which user to target:
    used_user_label = None
    if args.user is not None and args.user in u2i:
        uidx = int(u2i[args.user])
        used_user_label = args.user
    else:
        # If no --user given or not found, pick first non-empty user
        nz = np.where(mat.getnnz(axis=1) > 0)[0]
        if args.user is not None and args.user not in u2i:
            print(f"[info] user '{args.user}' not found in CSV; using a non-empty user instead.")
        if nz.size:
            uidx = int(nz[0])
            used_user_label = str(users[uidx])
        else:
            # extreme edge case: dataset empty -> popularity only
            print("[warn] no non-empty users; returning popularity top-K.")
            print("Top-K (popularity):", popularity[:args.k])
            return

    # 3) Train ALS (user × item, no transpose for your env)
    model = train_als(mat)

    # 4) Recommend for that user
    item_ids, scores, used_uidx = recommend_topn(model, uidx, mat, N=args.k)

    # 5) Map model item indices -> game names, aligned to matrix columns
    n_mat_items = mat.shape[1]
    inv_i2i = {int(idx): str(name) for name, idx in i2i.items()}

    def id_to_name(i):
        j = int(i)
        if 0 <= j < n_mat_items:
            return inv_i2i.get(j, f"[unknown:{j}]")
        return f"[out_of_range:{j}/{n_mat_items}]"

    names = [id_to_name(i) for i in item_ids]
    pretty = [f"{n} (score={s:.3f})" for n, s in zip(names, scores)]

    print("User (internal index):", used_uidx)
    print("User (original id):", used_user_label)
    print(f"Top-{args.k}:", pretty)

if __name__ == "__main__":
    main()
