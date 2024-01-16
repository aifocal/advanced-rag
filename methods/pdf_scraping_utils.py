import camelot
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# Still buggy and needs fixes
def extract_tables_from_pdf(file_path):
    tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')

    if len(tables) == 0:
        tables = camelot.read_pdf(file_path, pages='all', flavor='stream')

    for i, table in enumerate(tables):
        print(f"Table {i+1}:")
        print(table.df) 

    return tables


if __name__ == "__main__":
    # pdf_file_path = "data/pizzapizza.pdf"
    pdf_file_path = "data/pizza73_ingredient_list.pdf"
    pdf_text = extract_text_from_pdf(pdf_file_path)
    print(pdf_text)