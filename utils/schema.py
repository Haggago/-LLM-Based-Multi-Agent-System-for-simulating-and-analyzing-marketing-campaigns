"""
Schema definition for the raw Google-Maps-scraped dataset.

The scraper exports human-readable column headers (with spaces). We map every
raw column to a normalized snake_case name so the rest of the codebase never
has to deal with 'STREET ADDRESS'-style strings.

If your export has slightly different column names, edit RAW_TO_NORMALIZED
below -- nothing else in the pipeline needs to change.
"""

from __future__ import annotations

# raw_header -> normalized_name
RAW_TO_NORMALIZED: dict[str, str] = {
    "MATCH FILTERS": "match_filters",
    "NO MATCH REASONS": "no_match_reasons",
    "NAME": "name",
    "CATEGORY": "category",
    "STREET ADDRESS": "street_address",
    "ADDRESS": "address",
    "CITY": "city",
    "REGION": "region",
    "ZIP CODE": "zip_code",
    "COUNTRY NAME": "country_name",
    "COUNTRY CODE": "country_code",
    "PHONE": "phone",
    "EMAIL": "email",
    "EMAIL STATUS": "email_status",
    "EMAIL VERIFIED AT": "email_verified_at",
    "WEBSITE": "website",
    "FACEBOOK": "facebook",
    "INSTAGRAM": "instagram",
    "URL": "url",
    "BOOKING LINK": "booking_link",
    "LAT": "lat",
    "LNG": "lng",
    "SCORE": "score",
    "RATINGS": "ratings",
    "IS TEMPORARILY CLOSED": "is_temporarily_closed",
    "IS PERMANENTLY CLOSED": "is_permanently_closed",
    "PRICE": "price",
    "OPENING HOURS": "opening_hours",
    "POPULAR TIMES": "popular_times",
    "MAIN IMAGE URL": "main_image_url",
    "IMAGES COUNT": "images_count",
    "DESCRIPTION": "description",
    "HAS": "has_flag_raw",  # ambiguous truncated header in source export
    "OWNER ID": "owner_id",
    "PLACE ID": "place_id",
    "CID": "cid",
    "ZERO X OBJECT": "zero_x_object",
    "RESULT POSITION": "result_position",
    "TASK ID": "task_id",
    "COLLECTED AT": "collected_at",
    "INPUT URL": "input_url",
    "INPUT CITY": "input_city",
    "INPUT REGION": "input_region",
    "INPUT COUNTRY": "input_country",
    "INPUT CATEGORY": "input_category",
    "INPUT DISTRICT": "input_district",
    "PARAM LANGUAGE": "param_language",
    "PARAM DETAILS": "param_details",
    "PARAM COLLECT CONTACTS": "param_collect_contacts",
    "PARAM RATINGS": "param_ratings",
    "PARAM COUNTRY": "param_country",
    "PARAM IMAGES": "param_images",
    "PARAM MAX RESULTS": "param_max_results",
}

# Columns we treat as booleans after normalization
BOOLEAN_COLUMNS = [
    "is_temporarily_closed",
    "is_permanently_closed",
    "param_collect_contacts",
    "param_ratings",
    "param_images",
]

# Columns we coerce to numeric
NUMERIC_COLUMNS = [
    "lat", "lng", "score", "ratings", "images_count",
    "result_position", "param_max_results",
]

# The best available unique identifier(s) for entity resolution, in priority order
ID_COLUMNS_PRIORITY = ["place_id", "cid"]

# Fields used to build a fuzzy-match blocking key when place_id/cid are missing
FUZZY_MATCH_FIELDS = ["name", "street_address", "zip_code", "phone"]


def normalize_columns(columns) -> dict[str, str]:
    """Return a rename map for whatever columns are actually present in a dataframe.

    Falls back to a slugified version of the header for any column we don't
    explicitly know about, so unexpected extra columns don't crash ingestion.
    """
    rename_map = {}
    for col in columns:
        if col in RAW_TO_NORMALIZED:
            rename_map[col] = RAW_TO_NORMALIZED[col]
        else:
            rename_map[col] = (
                col.strip().lower().replace(" ", "_").replace("-", "_")
            )
    return rename_map
