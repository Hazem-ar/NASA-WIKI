import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
import re
from pathlib import Path
from urllib.parse import urljoin



def web_scraper(csv_input_file):
    print('_' * 40)
    print(f'[INFO] Scraping Started on urls from : {csv_input_file}')

    # Read csv
    df = pd.read_csv(csv_input_file)

    # Ready the session for scraping
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive"
    }

    session = requests.Session()
    session.headers.update(headers)

    scrap_time = time.time()

    names, urls = df['Name'], df['URL']
    size = len(names)
    df['Content'] = pd.NA   # this will store the single string (text + images)
    data_name = 'Nasa_Data'

    for i in range(size):
        url = urls[i]
        name = names[i]
        print(f'[ INFO: {i + 1}/{size}] Visiting {name} at {url}')

        try:
            r = session.get(url, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')

            for tag in soup(['header', 'footer', 'nav', 'aside', 'script', 'style']):
                tag.decompose()

            container = soup.find("article") or soup.body
            parts = []

            for elem in container.descendants:
                if elem.name == "p":
                    text = elem.get_text(strip=True)
                    if text:
                        parts.append(text)
                elif elem.name == "img" and elem.has_attr("src"):
                    img_url = urljoin(url, elem["src"])
                    parts.append(f"[IMAGE: {img_url}]")
            content = "\n".join(parts)

            # if len(content) >= 500:  # filter very short results
            df.at[i, 'Content'] = content
            # df.at[i, 'HTML'] = str(soup)

            print(f'[INFO: {i + 1}/{size}] Scraped {len(content)} chars from {url}')

        except Exception as e:
            print(f'[INFO: {i + 1}/{size}] Error scraping {url}: {e}')

    # Drop rows with empty content
    df = df.dropna(subset=['Content'])

    ROOT = Path(__file__).resolve().parent.parent
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    output_file = f'{data_name}_scraped_data.csv'
    output_file = data_dir / output_file

    df.to_csv(output_file, index=False)
    bank_end = time.time() - scrap_time
    print(f'[INFO] Saved Data in {output_file}')
    print(f'[INFO] Finished Script. Time Taken {bank_end:.2f}s')
    print('_' * 40)

    return output_file
