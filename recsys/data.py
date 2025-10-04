# recsys/data.py
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

def load_steam200k(fname: str = "steam-200k.csv") -> pd.DataFrame:
    """
    Loads Steam-200k (user, game, action, value, other)
    -> returns tidy implicit-feedback table: [user, game, strength]
    """
    df = pd.read_csv(DATA / fname, header=None)
    df.columns = ["user", "game", "action", "value", "other"]

    # keep only positive signals (play/purchase)
    df = df[df["action"].isin(["play", "purchase"])].copy()

    # strength = hours if 'play', else small positive for 'purchase'
    def _strength(row):
        try:
            return float(row["value"]) if row["action"] == "play" else 1.0
        except Exception:
            return 1.0

    df["strength"] = df.apply(_strength, axis=1)

    # cast & basic cleaning
    df = df.dropna(subset=["user", "game", "strength"])
    df["user"] = df["user"].astype(str)
    df["game"] = df["game"].astype(str)
    df["strength"] = df["strength"].clip(lower=0).astype("float32")

    # (optional) aggregate duplicate (user, game) rows by summing strength
    df = df.groupby(["user", "game"], as_index=False, sort=False)["strength"].sum()

    return df[["user", "game", "strength"]]
