import nbformat
from nbconvert import PDFExporter

def save_as_pdf(notebook_filename, pdf_filename):
    # Leggi il notebook
    with open(notebook_filename, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Esporta in PDF
    pdf_exporter = PDFExporter()
    pdf_data, _ = pdf_exporter.from_notebook_node(nb)

    # Salva il PDF
    with open(pdf_filename, 'wb') as f:
        f.write(pdf_data)

    print(f"PDF salvato come: {pdf_filename}")
