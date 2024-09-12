# Written by GPT-4
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Ensure the path to the tesseract executable is set if it's not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'<path_to_your_tesseract_executable>'


def convert_pdf_to_text(pdf_path, output_txt_path):
    # Open the PDF file
    document = fitz.open(pdf_path)

    text = ""  # Initialize a text string to hold all text from the PDF

    for page_num in range(len(document)):
        # Get the page
        page = document.load_page(page_num)

        # First, try to extract text using PyMuPDF
        text_content = page.get_text()

        if text_content.strip():  # If text is found, append it.
            text += text_content
    # Close the document
    # TODO add context fixing by replacing "you" with "I" in user messages, and "your" with "my" in user messages. Maybe some others.
    document.close()

    # Write the text to a .txt file
    with open(output_txt_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)

    return text


# Usage
pdf_path = "./Introduction to Logic and Critical Thinking, by Matthew Van Cleave.pdf"
output_txt_path = (
    "Introduction to Logic and Critical Thinking, by Matthew Van Cleave.txt"
)
convert_pdf_to_text(pdf_path, output_txt_path)
