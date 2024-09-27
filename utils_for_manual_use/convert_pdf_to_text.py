# Written by GPT-4
import fitz  # PyMuPDF
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
        # else:
        #     # If no text is found, it might be an image-based PDF
        #     # Extract the image from the page
        #     for img_index, img in enumerate(page.get_images(full=True)):
        #         xref = img[0]
        #         base_image = document.extract_image(xref)
        #         image_bytes = base_image["image"]

        #         # Load it to PIL
        #         image = Image.open(io.BytesIO(image_bytes))

        #         # Use pytesseract to do OCR on the image
        #         text += pytesseract.image_to_string(image)

    # Close the document
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
