import pandas as pd
import pytest

from src.pipeline.clean import CleaningPipeline, normalize_category_rule_based, validate_email


def test_validate_email_valid():
    email, valid = validate_email("Test@Example.com")
    assert valid is True
    assert email == "test@example.com"


def test_validate_email_invalid():
    email, valid = validate_email("not-an-email")
    assert valid is False
    assert email is None


def test_validate_email_none():
    email, valid = validate_email(None)
    assert valid is False
    assert email is None


def test_normalize_category_rule_based():
    assert normalize_category_rule_based("  drug rehab center  ") == "Drug Rehab Center"
    assert normalize_category_rule_based(None) is None


def test_cleaning_pipeline_drops_short_names():
    config = {"cleaning": {"min_name_length": 3}}
    df = pd.DataFrame({
        "name": ["AB", "Valid Business Name", None],
        "score": [4.5, 5.0, 3.0],
        "lat": [52.5, 53.5, 91.0],  # last one invalid (>90)
        "phone": [None, None, None],
        "email": [None, None, None],
        "category": ["rehab", "clinic", "clinic"],
    })
    result = CleaningPipeline(config).run(df)
    assert len(result) == 1
    assert result.iloc[0]["name"] == "Valid Business Name"


def test_cleaning_pipeline_clamps_invalid_lat():
    config = {"cleaning": {"min_name_length": 1}}
    df = pd.DataFrame({
        "name": ["Business A"],
        "lat": [95.0],
        "score": [4.0],
        "phone": [None],
        "email": [None],
        "category": ["rehab"],
    })
    result = CleaningPipeline(config).run(df)
    assert pd.isna(result.iloc[0]["lat"])
