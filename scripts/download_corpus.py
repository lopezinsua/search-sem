"""
Download AI/ML paper abstracts from arXiv API (free, no auth required).
Filters: cs.CL (NLP), cs.AI (AI), cs.LG (Machine Learning), cs.IR (Information Retrieval)
Usage: python scripts/download_corpus.py --limit 2000 --out data/corpus/
"""
import argparse
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
CATEGORIES = "cat:cs.CL OR cat:cs.AI OR cat:cs.LG OR cat:cs.IR"
BATCH = 200
ATOM_NS = "http://www.w3.org/2005/Atom"


def fetch_batch(start: int, max_results: int) -> list[dict]:
    params = urllib.parse.urlencode({
        "search_query": CATEGORIES,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        xml = resp.read()

    root = ET.fromstring(xml)
    papers = []
    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        title_el = entry.find(f"{{{ATOM_NS}}}title")
        abstract_el = entry.find(f"{{{ATOM_NS}}}summary")
        if title_el is None or abstract_el is None:
            continue
        title = " ".join(title_el.text.split())
        abstract = " ".join(abstract_el.text.split())
        if len(abstract.split()) < 30:
            continue
        papers.append({"title": title, "abstract": abstract})
    return papers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--out", default="data/corpus")
    parser.add_argument("--no-delay", action="store_true", help="Skip politeness delay (for CI/Docker builds)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_papers: list[dict] = []
    start = 0

    print("Downloading arXiv AI/ML abstracts (cs.CL, cs.AI, cs.LG, cs.IR)...")
    print(f"Target: {args.limit} papers\n")

    while len(all_papers) < args.limit:
        batch_size = min(BATCH, args.limit - len(all_papers))
        try:
            papers = fetch_batch(start, batch_size)
        except Exception as e:
            print(f"  Error at offset {start}: {e} — retrying in 5s")
            time.sleep(5)
            continue

        if not papers:
            break

        all_papers.extend(papers)
        print(f"  Fetched {len(all_papers)}/{args.limit}")
        start += BATCH
        if not args.no_delay:
            time.sleep(3)  # be polite with arXiv API

    # Save as one .txt per 500 papers (abstract per line, separated by ---)
    chunk_size = 500
    for i in range(0, len(all_papers), chunk_size):
        batch = all_papers[i : i + chunk_size]
        lines = []
        for p in batch:
            lines.append(f"{p['title']}. {p['abstract']}")
        file_idx = i // chunk_size
        path = out / f"arxiv_{file_idx:04d}.txt"
        path.write_text("\n\n---\n\n".join(lines), encoding="utf-8")
        print(f"  Saved {path.name} ({len(batch)} abstracts)")

    print(f"\nDone. {len(all_papers)} papers saved to {out}/")


if __name__ == "__main__":
    main()
