import os
from pypdf import PdfReader
import fitz  # PyMuPDF
from PIL import Image
import pytesseract


def convert_pdf_to_text(pdf_path, output_folder):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}.txt")

    if os.path.exists(output_path):
        print(f"Skipping already converted file: {output_path}")
        return output_path

    try:
        # Try to extract text directly
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    # Try different encodings if UTF-8 fails
                    encodings = ["utf-8", "latin-1", "ascii", "utf-16"]
                    for encoding in encodings:
                        try:
                            text += page_text.encode(encoding).decode("utf-8") + "\n"
                            break
                        except UnicodeEncodeError:
                            continue
                        except UnicodeDecodeError:
                            continue
                except Exception as e:
                    print(f"Error extracting text from page in {pdf_path}: {str(e)}")
                    continue  # Skip this page and continue with the next

        if text.strip():
            with open(output_path, "w", encoding="utf-8", errors="ignore") as out_file:
                out_file.write(text)
            return output_path
    except Exception as e:
        print(f"Error in direct text extraction for {pdf_path}: {str(e)}")
        # If direct extraction fails, proceed to OCR

    # Use OCR for scanned PDFs
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            try:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img)
                text += page_text + "\n"
            except Exception as e:
                print(f"Error processing page in {pdf_path}: {str(e)}")
                continue  # Skip this page and continue with the next

        with open(output_path, "w", encoding="utf-8", errors="ignore") as out_file:
            out_file.write(text)
        return output_path
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return None
