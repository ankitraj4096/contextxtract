from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re

def clean_text(text):
    # Remove unnecessary whitespaces, newlines, tabs
    return re.sub(r'\s+', ' ', text).strip()

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ''
    for page in reader.pages:
        raw_text += page.extract_text() + ' '
    return clean_text(raw_text)

def create_pdf(text, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    margin = 40
    line_height = 14
    y = height - margin

    words = text.split(' ')
    line = ""
    for word in words:
        if c.stringWidth(line + word + ' ') < (width - 2 * margin):
            line += word + ' '
        else:
            c.drawString(margin, y, line.strip())
            y -= line_height
            line = word + ' '
            if y < margin:
                c.showPage()
                y = height - margin
    if line:
        c.drawString(margin, y, line.strip())
    c.save()


pdf_input_path = "Arogya Sanjeevani Policy.pdf"
pdf_output_path = "cleaned_text_output.pdf"


text = extract_text_from_pdf(pdf_input_path)
create_pdf(text, pdf_output_path)

print("Cleaned text PDF created:", pdf_output_path)
