#!/usr/bin/env python3
"""
Preprocessing module for Multilingual Entity Resolution.
Normalizes Unicode (NFKC), removes structural punctuation/symbols,
cleans corporate suffixes/noise, and constructs composite record texts.
"""

import re
import unicodedata
import pandas as pd

# Regular expressions for corporate suffixes/noise
# Matches word boundaries so names like 'Lincoln' don't get modified by 'Inc'
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
    r'\b(एलएलपी|प्राइवेट\s+लिमिटेड|लिमिटेड)\b',  # Indic corporate tokens in Devanagari
]

CORP_REGEX = re.compile('|'.join(CORP_PATTERNS), flags=re.IGNORECASE)

# Leading / trailing non-alphanumeric/non-Indic structural symbols
LEADING_TRAILING_NOISE = re.compile(r'^[^\w\u0900-\u097F\u0B80-\u0BFF\u0C80-\u0CFF\u0600-\u06FF]+|[^\w\u0900-\u097F\u0B80-\u0BFF\u0C80-\u0CFF\u0600-\u06FF]+$', re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize text using NFKC, strip leading/trailing structural symbols, preserve Indic/French chars."""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # NFKC Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Replace whitespace characters with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Strip leading/trailing structural non-word symbols like --, ##, **, ,, .
    text = LEADING_TRAILING_NOISE.sub('', text).strip()
    
    return text


def clean_business_name(name: str) -> str:
    """Clean corporate noise while retaining core entity tokens."""
    text = normalize_text(name)
    if not text:
        return ""
    
    # Strip corporate suffixes/noise
    cleaned = CORP_REGEX.sub('', text)
    cleaned = normalize_text(cleaned)
    
    # If cleaning removed everything, fallback to original normalized text
    return cleaned if cleaned else text


def clean_address(address: str) -> str:
    """Clean and normalize business address."""
    text = normalize_text(address)
    if not text:
        return ""
    
    # Standardize commas and spaces
    text = re.sub(r'[\t\r\n]+', ' ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_composite_text(clean_name: str, clean_address: str) -> str:
    """Construct composite text string for embeddings."""
    return f"Name: {clean_name} | Address: {clean_address}"


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing pipeline to entity DataFrame."""
    df = df.copy()
    
    # Fill missing values
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('UNKNOWN').astype(str).str.strip().str.upper()
    
    df['clean_name'] = df['business_name'].apply(clean_business_name)
    df['clean_address'] = df['business_address'].apply(clean_address)
    df['composite_text'] = df.apply(
        lambda r: create_composite_text(r['clean_name'], r['clean_address']), axis=1
    )
    return df
