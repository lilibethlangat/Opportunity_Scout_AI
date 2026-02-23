import requests
import pandas as pd
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# -----------------------------
# CONFIGURATION
# -----------------------------
# -----------------------------
# CONFIGURATION
# -----------------------------
# Using the yc-oss mirror which is reachable and updated daily
MIRROR_URL = "https://yc-oss.github.io/api/companies/all.json"

TARGET_COUNT = 1000

# Using local paths relative to the script
DATA_DIR = Path("./data")
RAW_OUTPUT_PATH = DATA_DIR / "raw" / "yc_raw.json"
CSV_OUTPUT_PATH = DATA_DIR / "processed" / "yc_raw.csv"
LOG_PATH = Path("./logs") / "yc_scrape.log"


# -----------------------------
# LOGGING SETUP
# -----------------------------
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting YC scrape process via yc-oss mirror")


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def extract_year_from_batch(batch):
    """
    Converts batch like 'W20' or 'S18' into year 2020 or 2018
    """
    if not batch or len(batch) < 2:
        return None
    year_suffix = batch[1:]
    if len(year_suffix) == 1:
        prefix = "200"
    else:
        prefix = "20"
    try:
        return int(prefix + year_suffix)
    except ValueError:
        return None


# -----------------------------
# MAIN SCRAPER
# -----------------------------
def scrape_yc(limit=TARGET_COUNT):
    print(f"Fetching companies from mirror: {MIRROR_URL}...")
    
    try:
        response = requests.get(MIRROR_URL, timeout=30)
        response.raise_for_status()
        all_companies = response.json()
        
        count = len(all_companies)
        print(f"Successfully fetched {count} companies from mirror.")
        
        if count > limit:
            print(f"Limiting to first {limit} companies as requested.")
            return all_companies[:limit]
        return all_companies
        
    except Exception as e:
        logging.error(f"Failed to fetch from mirror: {e}")
        print(f"Error: {e}")
        return []


# -----------------------------
# PROCESS JSON TO DATAFRAME
# -----------------------------
def transform_to_dataframe(companies):
    rows = []

    for company in companies:
        # Map fields from the mirror format
        row = {
            "name": company.get("name"),
            "batch": company.get("batch"),
            "founded_year": extract_year_from_batch(company.get("batch")),
            "one_liner": company.get("one_liner"),
            "industries": ", ".join(company.get("industries", [])),
            "regions": ", ".join(company.get("regions", [])),
            "company_url": company.get("yc_url") or f"https://www.ycombinator.com/companies/{company.get('slug')}",
            "website": company.get("website"),
            "status": company.get("status"),
            "description": company.get("long_description")
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


# -----------------------------
# SAVE FUNCTIONS
# -----------------------------
def save_raw_json(companies):
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_OUTPUT_PATH, "w") as f:
        json.dump(companies, f, indent=4)


def save_csv(df):
    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUTPUT_PATH, index=False)


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    start_time = datetime.now()
    logging.info("Scraping started")

    companies = scrape_yc(TARGET_COUNT)

    logging.info(f"Total companies fetched: {len(companies)}")

    save_raw_json(companies)

    df = transform_to_dataframe(companies)
    save_csv(df)

    logging.info("Scraping completed successfully")
    logging.info(f"Execution time: {datetime.now() - start_time}")

    print(f"\nSuccess! Scraping completed.")
    print(f"Total companies fetched: {len(companies)}")
    print(f"Data saved to {CSV_OUTPUT_PATH}")
