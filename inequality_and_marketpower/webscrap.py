"""
Bengali Wikipedia Corpus 

Usage:
    pip install requests tqdm
    python webscrap.py

Output:
    - inequality_and_marketpower/bengali_wiki_articles/
    - inequality_and_marketpower/bengali_wiki_corpus.jsonl
    - inequality_and_marketpower/corpus_stats.txt
"""

import requests
import json
import os
import time
import re
from tqdm import tqdm

TARGET_ARTICLES   = 500
LANGUAGE          = "bn"
API_URL           = f"https://{LANGUAGE}.wikipedia.org/w/api.php"
BASE_DIR          = "/Users/shamsimumtahinamomo/Documents/111A- Spring 2026/Data economics/inequality_and_marketpower"
OUTPUT_DIR        = os.path.join(BASE_DIR, "bengali_wiki_articles")
CORPUS_FILE       = os.path.join(BASE_DIR, "bengali_wiki_corpus.jsonl")
STATS_FILE        = os.path.join(BASE_DIR, "corpus_stats.txt")
MIN_WORD_COUNT    = 10
BATCH_SIZE        = 50
SLEEP_BETWEEN     = 0.5

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "BengaliWikiCorpus/1.0 (student NLP research project; contact: student@brandeis.edu)"
})

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(raw: str) -> str:
    raw = re.sub(r'\{\{[^}]*\}\}', '', raw)
    raw = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', raw)
    raw = re.sub(r'<[^>]+>', '', raw)
    raw = re.sub(r'={2,}[^=]+=+', '', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    return raw.strip()


def get_random_article_titles(n: int) -> list[str]:
    """Keep fetching until we have n titles — no cap."""
    titles = []
    params = {
        "action":      "query",
        "list":        "random",
        "rnnamespace": 0,
        "rnlimit":     500,
        "format":      "json",
    }
    print("Fetching random article titles...")
    while len(titles) < n:
        try:
            resp = SESSION.get(API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            batch = [item["title"] for item in data["query"]["random"]]
            titles.extend(batch)
            time.sleep(SLEEP_BETWEEN)
            print(f"  {len(titles)} titles collected...")
        except Exception as e:
            print(f"  Warning fetching titles: {e} -- retrying")
            time.sleep(3)
    return titles[:n]


def fetch_articles_batch(titles: list[str]) -> list[dict]:
    post_data = {
        "action":          "query",
        "titles":          "|".join(titles),
        "prop":            "extracts",
        "explaintext":     "1",
        "exsectionformat": "plain",
        "format":          "json",
        "formatversion":   "2",
    }
    resp = SESSION.post(API_URL, data=post_data, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    articles = []
    for page in result["query"]["pages"]:
        if page.get("missing") or "extract" not in page:
            continue
        text = clean_text(page["extract"])
        word_count = len(text.split())
        if word_count < MIN_WORD_COUNT:
            continue
        articles.append({
            "id":         page["pageid"],
            "title":      page["title"],
            "text":       text,
            "word_count": word_count,
        })
    return articles


def save_article_txt(article: dict):
    safe_title = re.sub(r'[^\w\u0980-\u09FF\s-]', '', article["title"])[:80]
    filename = f"{article['id']}_{safe_title}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {article['title']}\n")
        f.write(f"ID: {article['id']}\n")
        f.write(f"WORDS: {article['word_count']}\n")
        f.write("=" * 60 + "\n\n")
        f.write(article["text"])


def compute_stats(articles: list[dict]) -> dict:
    word_counts = [a["word_count"] for a in articles]
    total_words = sum(word_counts)
    return {
        "total_articles": len(articles),
        "total_words":    total_words,
        "avg_words":      round(total_words / len(articles), 1) if articles else 0,
        "min_words":      min(word_counts) if word_counts else 0,
        "max_words":      max(word_counts) if word_counts else 0,
        "median_words":   sorted(word_counts)[len(word_counts)//2] if word_counts else 0,
    }


def main():
    print("=" * 60)
    print("  Bengali Wikipedia Corpus Downloader")
    print(f"  Target: {TARGET_ARTICLES} articles")
    print(f"  Saving to: {BASE_DIR}")
    print("=" * 60)

    collected = []
    corpus_out = open(CORPUS_FILE, "w", encoding="utf-8")
    fetched_titles = []
    title_index = 0

    print(f"\nDownloading article content in batches of {BATCH_SIZE}...")
    with tqdm(total=TARGET_ARTICLES, unit="article") as pbar:
        while len(collected) < TARGET_ARTICLES:

            if title_index >= len(fetched_titles) - BATCH_SIZE:
                new_titles = get_random_article_titles(500)
                fetched_titles.extend(new_titles)

            batch_titles = fetched_titles[title_index:title_index + BATCH_SIZE]
            title_index += BATCH_SIZE

            try:
                articles = fetch_articles_batch(batch_titles)
                for article in articles:
                    if len(collected) >= TARGET_ARTICLES:
                        break
                    collected.append(article)
                    save_article_txt(article)
                    corpus_out.write(json.dumps(article, ensure_ascii=False) + "\n")
                    pbar.update(1)
            except requests.RequestException as e:
                print(f"\n  Warning: {e} -- retrying after 5s")
                time.sleep(5)
            time.sleep(SLEEP_BETWEEN)

    corpus_out.close()

    stats = compute_stats(collected)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        f.write("Bengali Wikipedia Corpus -- Statistics\n")
        f.write("=" * 40 + "\n")
        for k, v in stats.items():
            f.write(f"{k:20s}: {v:,}\n")

    print("\n" + "=" * 60)
    print("  Download Complete!")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:20s}: {v:,}")
    print(f"\n  Individual articles -> {OUTPUT_DIR}")
    print(f"  Corpus JSONL file   -> {CORPUS_FILE}")
    print(f"  Stats               -> {STATS_FILE}")


if __name__ == "__main__":
    main()