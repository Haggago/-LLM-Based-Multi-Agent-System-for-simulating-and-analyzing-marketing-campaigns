"""Generates a small synthetic dataset matching the real export's schema, so
the pipeline can be run and tested end-to-end without the actual 2.68M-row file.

Usage:
    python data/sample/generate_sample.py --rows 2000 --out data/raw/businesses.csv
"""

from __future__ import annotations

import argparse
import random
import string
from datetime import datetime, timedelta

import pandas as pd

random.seed(42)

CATEGORIES_RAW = [
    "Drug rehabilitation center", "Addiction treatment center", "Rehab Clinic",
    "drug treatment center", "Methadone clinic", "Detox center", "Sober living home",
    "Counseling center", "Mental health service", "Alcohol treatment center",
    "Substance abuse treatment center", "Health Consultant", "Hospital", "Clinic",
    "Support group", "Non-profit organization", "", "Rehabilitation Center",
]

CITIES = [
    ("Hamburg", "Hamburg", "20095", "DE"), ("Berlin", "Berlin", "10115", "DE"),
    ("Munich", "Bavaria", "80331", "DE"), ("Cologne", "North Rhine-Westphalia", "50667", "DE"),
    ("Frankfurt", "Hesse", "60311", "DE"), ("Stuttgart", "Baden-Württemberg", "70173", "DE"),
]

FIRST_WORDS = ["Hope", "New Life", "Serenity", "Recovery", "Bridges", "Pathways", "Renewal", "Horizon"]
SECOND_WORDS = ["Rehab Center", "Treatment Clinic", "Recovery House", "Wellness Center", "Health Group"]


def rand_str(n=8):
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def make_row(i: int) -> dict:
    city, region, zip_code, country_code = random.choice(CITIES)
    name = f"{random.choice(FIRST_WORDS)} {random.choice(SECOND_WORDS)}"
    has_place_id = random.random() > 0.05  # 5% missing, to exercise fuzzy dedup path
    is_dupe = random.random() < 0.1 and i > 0

    row = {
        "MATCH FILTERS": "true",
        "NO MATCH REASONS": "",
        "NAME": name,
        "CATEGORY": random.choice(CATEGORIES_RAW),
        "STREET ADDRESS": f"{random.randint(1,200)} Main St",
        "ADDRESS": f"{random.randint(1,200)} Main St, {city}",
        "CITY": city,
        "REGION": region,
        "ZIP CODE": zip_code,
        "COUNTRY NAME": "Germany",
        "COUNTRY CODE": country_code,
        "PHONE": f"+49 {random.randint(30,89)} {random.randint(1000000,9999999)}",
        "EMAIL": f"{rand_str(6).lower()}@example.com" if random.random() > 0.3 else "",
        "EMAIL STATUS": random.choice(["verified", "unverified", ""]),
        "EMAIL VERIFIED AT": "",
        "WEBSITE": f"https://{rand_str(6).lower()}.de" if random.random() > 0.4 else "",
        "FACEBOOK": f"https://facebook.com/{rand_str(6)}" if random.random() > 0.6 else "",
        "INSTAGRAM": f"https://instagram.com/{rand_str(6)}" if random.random() > 0.7 else "",
        "URL": f"https://maps.google.com/?cid={i}",
        "BOOKING LINK": "",
        "LAT": round(50 + random.random() * 5, 6),
        "LNG": round(8 + random.random() * 5, 6),
        "SCORE": round(random.uniform(1, 5), 1),
        "RATINGS": random.randint(0, 300),
        "IS TEMPORARILY CLOSED": random.random() < 0.02,
        "IS PERMANENTLY CLOSED": random.random() < 0.03,
        "PRICE": random.choice(["", "$", "$$", "$$$"]),
        "OPENING HOURS": '{"mon": "9-17", "tue": "9-17", "wed": "9-17"}' if random.random() > 0.3 else "",
        "POPULAR TIMES": "",
        "MAIN IMAGE URL": f"https://images.example.com/{rand_str(6)}.jpg" if random.random() > 0.5 else "",
        "IMAGES COUNT": random.randint(0, 40),
        "DESCRIPTION": (
            f"{name} offers comprehensive treatment and support services for individuals "
            f"seeking recovery in {city}." if random.random() > 0.2 else ""
        ),
        "HAS": "",
        "OWNER ID": rand_str(10),
        "PLACE ID": (f"PLACE_{i}" if not is_dupe else f"PLACE_{max(0,i-1)}") if has_place_id else "",
        "CID": str(random.randint(10**15, 10**16)),
        "ZERO X OBJECT": "",
        "RESULT POSITION": random.randint(1, 20),
        "TASK ID": rand_str(8),
        "COLLECTED AT": (datetime(2026, 1, 1) + timedelta(days=random.randint(0, 180))).isoformat(),
        "INPUT URL": "",
        "INPUT CITY": city,
        "INPUT REGION": region,
        "INPUT COUNTRY": "Germany",
        "INPUT CATEGORY": "drug rehab",
        "INPUT DISTRICT": "",
        "PARAM LANGUAGE": "en",
        "PARAM DETAILS": "true",
        "PARAM COLLECT CONTACTS": "true",
        "PARAM RATINGS": "true",
        "PARAM COUNTRY": country_code,
        "PARAM IMAGES": "true",
        "PARAM MAX RESULTS": 1000,
    }
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--out", default="data/raw/businesses.csv")
    args = p.parse_args()

    rows = [make_row(i) for i in range(args.rows)]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} synthetic rows to {args.out}")


if __name__ == "__main__":
    main()
