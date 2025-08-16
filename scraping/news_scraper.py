import requests
from bs4 import BeautifulSoup
import time, json, re
from itertools import count

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
        session.get(SWITCH_URL)  # set ke Indonesia
        # reload halaman
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
    images = [img["src"] for img in detail_div.find_all("img")]
    idx = next(id_counter)
    return {
        # "url": url,
        # "author": author,
        # "images": images
        "_id": f"cs_news_{idx}",
        "section": "Berita Ilmu Komputer",
        "title": title,
        "type": ["Berita"],
        "lang": "id",
        "text": f"Berita {title} tanggal : " + date + " " + content
    }

def scrape_list(page=0):
    url = BASE_URL.format(page if page else "")
    soup = ensure_indonesian(url)

    blocks = soup.select("div.col-sm-12.col-md-12.col-xs-12")
    for b in blocks:
        a = b.select_one("h4 a")
        if not a: 
            continue

        link = a["href"]
        thumb_style = b.select_one(".center-cropped-menejemen")["style"]
        thumb_url = re.search(r"url\('(.*?)'\)", thumb_style).group(1)

        short = b.select_one("div.col-sm-9 p").get_text(strip=True)

        try:
            detail = scrape_detail(link)
            # detail.update({
            #     "thumbnail": thumb_url,
            #     "excerpt": short
            # })
            all_news.append(detail)
            time.sleep(1)
        except Exception as e:
            print("Error detail:", link, e)


for page in range(0, 240, 10):
    print("Scraping page:", page)
    scrape_list(page)

with open("upi_news.json", "w", encoding="utf-8") as f:
    json.dump(all_news, f, ensure_ascii=False, indent=2)
