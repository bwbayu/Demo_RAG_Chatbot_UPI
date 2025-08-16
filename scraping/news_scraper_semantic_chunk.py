import requests
from bs4 import BeautifulSoup
import time, json, re
from itertools import count
import os
from dotenv import load_dotenv

# --- LangChain + Gemini Embeddings ---
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    model="gemini-embedding-001"
)
text_splitter = SemanticChunker(
    embeddings, breakpoint_threshold_type="gradient"
)

# --- Scraper setup ---
BASE_URL = "https://cs.upi.edu/v2/news_list/{}"
SWITCH_URL = "https://cs.upi.edu/v2/lang/set_language/ID"
session = requests.Session()
all_news = []
id_counter = count(1)

def check_language(soup):
    lang_el = soup.select_one("li.row-end a i")
    if not lang_el:
        return "unknown"
    text = lang_el.get_text(strip=True).lower()
    if "indonesia" in text:
        return "english"
    elif "english" in text:
        return "indonesia"
    return "unknown"

def ensure_indonesian(url):
    r = session.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    lang = check_language(soup)

    if lang == "english":
        session.get(SWITCH_URL)  # switch to Indonesian
        r = session.get(url)
        soup = BeautifulSoup(r.text, "html.parser")

    return soup

def scrape_detail(url):
    soup = ensure_indonesian(url)

    title = soup.find("h3").get_text(strip=True)
    meta = soup.find("h6").get_text(" ", strip=True)

    author = re.search(r"oleh:\s*(.*?)\s{2,}", meta)
    author = author.group(1) if author else None

    date = re.search(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}", meta)
    date = date.group(0) if date else None

    detail_div = soup.find("div", class_="news-detail")
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in detail_div.find_all("p", recursive=False)
    ]
    content = "\n".join(paragraphs)

    full_text = f"Berita {title} tanggal : {date}\n{content}"

    docs = text_splitter.create_documents([full_text])
    chunks = [d.page_content for d in docs]

    news_idx = next(id_counter)
    chunked_records = []

    for c_idx, chunk in enumerate(chunks, start=1):
        chunked_records.append({
            "_id": f"cs_news_{news_idx}_{c_idx}",
            "section": title + f" Group {c_idx}/{len(chunks)}",
            "title": "Berita Ilmu Komputer",
            "type": ["Berita"],
            "lang": "id",
            "text": chunk,
        })

    return chunked_records

def scrape_list(page=0):
    url = BASE_URL.format(page if page else "")
    soup = ensure_indonesian(url)

    blocks = soup.select("div.col-sm-12.col-md-12.col-xs-12")
    for b in blocks:
        a = b.select_one("h4 a")
        if not a:
            continue

        link = a["href"]

        try:
            detail = scrape_detail(link)
            all_news.extend(detail)
            time.sleep(1)
        except Exception as e:
            print("Error detail:", link, e)

for page in range(0, 10, 10):
    print("Scraping page:", page)
    scrape_list(page)

with open("upi_news.json", "w", encoding="utf-8") as f:
    json.dump(all_news, f, ensure_ascii=False, indent=2)
