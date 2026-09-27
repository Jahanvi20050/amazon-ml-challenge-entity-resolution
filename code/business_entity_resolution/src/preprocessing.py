#!/usr/bin/env python3
"""
Enhanced Preprocessing module for Multilingual Entity Resolution.
1. Normalizes Unicode (NFKC) preserving non-ASCII Indic (Devanagari, Tamil, Kannada) and French scripts.
2. Strips leading/trailing structural symbols and punctuation noise.
3. Cleans corporate suffix noise (dba, LLC, Pvt. Ltd., Inc, LLP, GmbH, etc.).
4. Extracts structured postal/zip codes and digit tokens from addresses.
5. Constructs enriched composite record texts for dense and sparse representations.
"""

import re
import unicodedata
import pandas as pd

# Corporate legal suffix regex
CORP_PATTERNS = [
    r'\b(pvt\.?\s*ltd\.?|private\s+limited)\b',
    r'\b(llc|l\.l\.c\.)\b',
    r'\b(inc\.?|incorporated)\b',
    r'\b(ltd\.?|limited)\b',
    r'\b(llp|l\.l\.p\.)\b',
    r'\b(corp\.?|corporation)\b',
    r'\b(co\.?|company)\b',
    r'\b(plc|gmbh|s\.a\.?|s\.r\.o\.?)\b',
    r'\b(dba|d\.b\.a\.)\b',
    r'\b(एलएलपी|प्राइवेट\s+लिमिटेड|लिमिटेड)\b',  # Indic corporate tokens
]

CORP_REGEX = re.compile('|'.join(CORP_PATTERNS), flags=re.IGNORECASE)

# Structural noise symbols
LEADING_TRAILING_NOISE = re.compile(
    r'^[^\w\u0900-\u097F\u0B80-\u0BFF\u0C80-\u0CFF\u0600-\u06FF]+|[^\w\u0900-\u097F\u0B80-\u0BFF\u0C80-\u0CFF\u0600-\u06FF]+$',
    re.UNICODE
)


def normalize_text(text: str) -> str:
    """Normalize text using NFKC Unicode, clean whitespace and structural symbols."""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = LEADING_TRAILING_NOISE.sub('', text).strip()
    return text


def clean_business_name(name: str) -> str:
    """Clean corporate noise while retaining core entity identity."""
    text = normalize_text(name)
    if not text:
        return ""
    
    cleaned = CORP_REGEX.sub('', text)
    cleaned = normalize_text(cleaned)
    return cleaned if cleaned else text


def clean_address(address: str) -> str:
    """Clean address string and standardize commas/spaces."""
    text = normalize_text(address)
    if not text:
        return ""
    
    text = re.sub(r'[\t\r\n]+', ' ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_postal_code(address: str) -> str:
    """Extract candidate postal/zip code (5-digit US zip or 6-digit India PIN code)."""
    if not isinstance(address, str):
        return ""
    # Search for 5-digit or 6-digit standalone numbers
    matches = re.findall(r'\b\d{5,6}\b', address)
    return matches[-1] if matches else ""


def create_composite_text(clean_name: str, clean_address: str) -> str:
    """Construct enriched composite text string for embeddings."""
    return f"Name: {clean_name} | Address: {clean_address}"


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply enhanced preprocessing pipeline to entity DataFrame."""
    df = df.copy()
    
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('UNKNOWN').astype(str).str.strip().str.upper()
    
    df['clean_name'] = df['business_name'].apply(clean_business_name)
    df['clean_address'] = df['business_address'].apply(clean_address)
    df['postal_code'] = df['clean_address'].apply(extract_postal_code)
    
    df['composite_text'] = df.apply(
        lambda r: create_composite_text(r['clean_name'], r['clean_address']), axis=1
    )
    return df
