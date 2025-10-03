import requests
import pandas as pd
from bs4 import BeautifulSoup
import csv
import re
from tqdm import tqdm  # <-- progress bar

INPUT_CSV = "SB_publication_PMC.csv"
OUTPUT_CSV = "articles_with_bodies.csv"
USER_AGENT = "Mozilla/5.0"

df = pd.read_csv(INPUT_CSV)

def scrape_article(link):
    try:
        resp = requests.get(link, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        body_section = soup.find("section", class_="body main-article-body")
        if not body_section:
            return ""

        # Remove references
        for bad in body_section.find_all("div", class_="ref-list"):
            bad.decompose()

        article_text = body_section.get_text("\n", strip=True)
        article_text = re.sub('\n', '', article_text)  # remove line breaks
        return article_text
    
    except Exception as e:
        print(f"Error scraping {link}: {e}")
        return ""
from tqdm import tqdm
tqdm.pandas()
# Wrap the iterable with tqdm for progress bar
results = df["Link"].progress_apply(scrape_article)  # <-- use progress_apply

# Save CSV
df['Text'] = results
df.to_csv(OUTPUT_CSV, index=False)

print(f"Done. Saved to {OUTPUT_CSV}")
