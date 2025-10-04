from Simple_Scraper import web_scraper
import pandas as pd
from pathlib import Path
from EDA import clean_data

documents_csv = 'all_links.csv'
directory = 'data'
ROOT = Path(__file__).resolve().parent.parent  # go to project root
CSV_PATH = ROOT / directory / documents_csv
scrapped = web_scraper(CSV_PATH)

eda = clean_data(scrapped)