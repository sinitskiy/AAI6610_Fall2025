import requests
import os
import time
import pandas as pd

# Settings
QUERY = "uncertainty prediction ML"
MAX_RESULTS = 1500  # Want among of PDF
YEAR_FROM = 2020
YEAR_TO = 2025
OUTPUT_DIR = "OA_articles"  # txt + PDF
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMAIL = os.environ.get("UNPAYWALL_EMAIL", "").strip()
UNPAYWALL_EMAIL = EMAIL
BATCH_SIZE = 50  # Each time fetch Crossref articles (can be larger than MAX_RESULTS)

def search_crossref(query, rows=50, offset=0, year_from=2020, year_to=2025):
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": rows,
        "offset": offset,
        "filter": f"from-pub-date:{year_from}-01-01,until-pub-date:{year_to}-12-31"
    }
    headers = {"User-Agent": f"Python script (mailto:{EMAIL})"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def safe_filename(s, maxlen=80):
    return "".join(c if c.isalnum() else "_" for c in s)[:maxlen]

def save_txt(meta, txt_path):
    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

def get_unpaywall_pdf(doi):
    url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if data.get("is_oa") and data.get("best_oa_location"):
                return data["best_oa_location"].get("url_for_pdf")
    except Exception as e:
        print("Unpaywall request error:", e)
    return None

def download_pdf(pdf_url, title):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(pdf_url, headers=headers, timeout=30)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            pdf_path = os.path.join(OUTPUT_DIR, f"{safe_filename(title)}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(r.content)
            return pdf_path
    except Exception as e:
        print("PDF download error:", e)
    return None

def main():
    csv_rows = []
    offset = 0
    success_count = 0

    while success_count < MAX_RESULTS:
        data = search_crossref(QUERY, rows=BATCH_SIZE, offset=offset, year_from=YEAR_FROM, year_to=YEAR_TO)
        items = data["message"]["items"]
        if not items:
            print("No more articles to fetch")
            break

        for item in items:
            if success_count >= MAX_RESULTS:
                break

            title = item.get("title", [""])[0]
            authors = ", ".join([f"{a.get('given','')} {a.get('family','')}" for a in item.get("author", [])])
            date_parts = item.get("issued", {}).get("date-parts", [[0]])
            published = "-".join(map(str, date_parts[0]))
            doi = item.get("DOI", "")
            abstract = item.get("abstract", "")

            # PDF check
            pdf_url = ""
            for link in item.get("link", []):
                if link.get("content-type") == "application/pdf":
                    pdf_url = link.get("URL")
                    break
            if not pdf_url and doi:
                pdf_url = get_unpaywall_pdf(doi)

            if not pdf_url:
                print(f"{success_count+1}) {title} | PDF: No")
                continue

            # Download PDF
            pdf_path = download_pdf(pdf_url, title)
            if not pdf_path:
                print(f"{success_count+1}) {title} | PDF can't download")
                continue

            # Save as txt
            meta = {
                "title": title,
                "authors": authors,
                "published": published,
                "doi": doi,
                "abstract": abstract,
                "pdf_url": pdf_url,
            }
            txt_path = os.path.join(OUTPUT_DIR, f"{safe_filename(title)}.txt")
            save_txt(meta, txt_path)

            # CSV
            csv_rows.append(meta)
            success_count += 1
            print(f"{success_count}) {title} | PDF: Yes")

            time.sleep(1)

        offset += BATCH_SIZE

    # Save as CSV
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        df.to_csv(os.path.join(OUTPUT_DIR, "articles_with_pdf.csv"), index=False, encoding="utf-8-sig")
        print(f"Successful！ Download{success_count}, PDFs at", OUTPUT_DIR)
    else:
        print("No PDF, CSV not generated。")

if __name__ == "__main__":
    main()
