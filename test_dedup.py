import pandas as pd

from src.pipeline.dedup import dedup_exact, dedup_fuzzy


def test_dedup_exact_keeps_most_recent():
    df = pd.DataFrame({
        "place_id": ["P1", "P1", "P2"],
        "cid": ["C1", "C1", "C2"],
        "collected_at": pd.to_datetime(["2026-01-01", "2026-03-01", "2026-02-01"]),
        "name": ["Old Record", "New Record", "Other"],
    })
    result = dedup_exact(df)
    assert len(result) == 2
    kept = result[result["place_id"] == "P1"].iloc[0]
    assert kept["name"] == "New Record"


def test_dedup_fuzzy_blocks_by_zip():
    df = pd.DataFrame({
        "place_id": [None, None, None],
        "cid": [None, None, None],
        "name": ["Hope Rehab Center", "Hope Rehab Centre", "Totally Different Name"],
        "street_address": ["1 Main St", "1 Main St", "9 Other Ave"],
        "zip_code": ["20095", "20095", "20095"],
    })
    result = dedup_fuzzy(df, block_by="zip_code", threshold=85)
    assert len(result) == 2  # the two near-identical rows collapse to one
