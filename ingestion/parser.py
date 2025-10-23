'''PDF parser placeholder'''
import pdfplumber   
from pdfminer.high_level import extract_text
import re

def parse_pdf(file_path):
    all_chunks = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages,start=1):
                text = page.extract_text() or ""
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    all_chunks.append({
                        "page_number": page_num,
                        "modality": "text",
                        "text": text
                    })
                
                tables = page.extract_tables()
                for tbl in tables:
                    if not tbl or len(tbl) < 2:
                        continue
                    rows = []
                    headers = tbl[0]
                    for row in tbl[1:]:
                        pairs = []
                        for h, v in zip(headers, row):
                            pairs.append(f"{h}: {v}")
                        if pairs:
                            rows.append(" | ".join(pairs))
                    if rows:
                        table_text = "table:\n" + "\n".join(rows)
                        all_chunks.append({
                            "page_number": page_num,
                            "modality": "table",
                            "text": table_text
                        })
    except Exception as e:
        print(f"Error parsing PDF: {e}")

    return all_chunks