## test pdf parser
from ingestion.parser import prase_pdf
if __name__ == "__main__":
    file_path = "soucepdf/AIA/accident/GrandVIP_sc.pdf"  # Replace with your PDF file path
    result = prase_pdf(file_path)
    print(result)
