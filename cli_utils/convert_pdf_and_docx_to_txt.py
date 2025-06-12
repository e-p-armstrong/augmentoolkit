#!/usr/bin/env python3
import os
import argparse
import io

# Necessary imports for PDF and DOCX processing, wrapped in a try-except block
try:
    from PIL import Image
    import pytesseract
    import fitz  # PyMuPDF
    import docx
except ImportError:
    print("--------------------------------------------------------------------")
    print("ERROR: Required libraries for PDF/DOCX processing are not installed.")
    print("Please install them to enable full functionality:")
    print("  pip install PyMuPDF Pillow pytesseract python-docx")
    print("You also need to have Tesseract OCR installed on your system.")
    print("Visit: https://tesseract-ocr.github.io/tessdoc/Installation.html")
    print("--------------------------------------------------------------------")

    # Define dummy functions if imports fail, so the script can still be loaded
    # but will raise an error if these functions are called.
    def extract_text_from_docx(path):
        raise ImportError("python-docx library is not installed.")

    def extract_text_from_pdf(path):
        raise ImportError("PyMuPDF library is not installed.")

    def extract_text_from_pdf_ocr(path):
        raise ImportError(
            "PyMuPDF, Pillow, or Pytesseract library is not installed, or Tesseract OCR is not set up."
        )

    def remove_newlines_in_sentences(text):
        return text  # No-op if other libraries failed

    def extract_text(path):
        raise ImportError(
            "One or more required libraries for file processing are missing."
        )


if "fitz" in globals():  # Check if PyMuPDF was imported successfully

    def extract_text_from_docx(path):
        """
        Extracts text from a DOCX file.
        """
        try:
            doc = docx.Document(path)
            full_text = [para.text for para in doc.paragraphs]
            return "\n".join(full_text)
        except Exception as e:
            print(f"Error processing DOCX file {path}: {e}")
            return ""

    def extract_text_from_pdf(path):
        """
        Extracts text from a copyable PDF using PyMuPDF.
        """
        text = ""
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:  # More general exception for fitz issues
            print(f"Warning: Skipping broken or unreadable PDF file {path}. Error: {e}")
            return ""
        return text

    def extract_text_from_pdf_ocr(path):
        """
        Extracts text from a non-copyable or poorly extracted PDF using OCR.
        """
        text = ""
        try:
            with fitz.open(path) as doc:
                for page_number, page in enumerate(doc):
                    try:
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_bytes))
                        page_text = pytesseract.image_to_string(img)
                        text += page_text + "\n"
                    except Exception as ocr_e:
                        print(
                            f"Error during OCR on page {page_number + 1} of {path}: {ocr_e}"
                        )
                        continue  # Try next page
        except Exception as e:
            print(
                f"Warning: Skipping broken or unreadable PDF during OCR for file {path}. Error: {e}"
            )
            return ""
        return text

    def remove_newlines_in_sentences(text):
        """
        Tries to join lines that are part of the same sentence.
        A simple approach: if a line doesn't end with punctuation,
        it's likely continued on the next line.
        """
        lines = text.split("\n")
        new_lines = []
        current_line_processed = False
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if (
                not stripped_line
            ):  # Keep empty lines if they were intentional double newlines
                if (
                    not current_line_processed and new_lines
                ):  # Avoid double newlines if previous was just added
                    new_lines.append("")
                current_line_processed = False
                continue

            if (
                new_lines
                and new_lines[-1].strip()
                and not new_lines[-1].strip()[-1] in ".!?"
            ):
                new_lines[-1] += " " + stripped_line
            else:
                new_lines.append(stripped_line)
            current_line_processed = True

        # Join with single newline, as multiple newlines would have been preserved as empty strings
        return "\n".join(new_lines)

    def extract_text(path):
        """
        Extracts formatted text from a PDF or DOCX file.
        """
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        text = ""

        if ext == ".docx":
            text = extract_text_from_docx(path)
        elif ext == ".pdf":
            text = extract_text_from_pdf(path)
            # If extracted text is too short (e.g. < 100 chars, might be scanned PDF), try OCR
            if len(text.strip()) < 100:
                print(
                    f"Short text from PDF {path} ({len(text.strip())} chars). Attempting OCR."
                )
                text_ocr = extract_text_from_pdf_ocr(path)
                if len(text_ocr.strip()) > len(
                    text.strip()
                ):  # Use OCR if it yields more text
                    text = text_ocr
                elif (
                    not text.strip() and text_ocr.strip()
                ):  # If initial was empty but OCR found something
                    text = text_ocr
        else:
            print(f"Unsupported file type: {ext} for file {path}")
            return ""  # Return empty string for unsupported types

        # The remove_newlines_in_sentences function was designed for text
        # where newlines might break sentences. Its effect might be disruptive
        # for some documents. Consider making its application optional or refining it.
        # For now, let's apply it as per the original chunking.py logic.
        # text = remove_newlines_in_sentences(text) # Commented out for now, can be too aggressive

        return text


def process_files(input_dir, output_dir):
    """
    Processes all .pdf and .docx files in the input directory and saves
    them as .txt files in the output directory, preserving structure.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith((".pdf", ".docx")):
                input_file_path = os.path.join(root, filename)

                # Determine output path
                relative_path = os.path.relpath(input_file_path, input_dir)
                output_filename = filename + ".txt"
                output_file_path = os.path.join(
                    output_dir, os.path.dirname(relative_path), output_filename
                )

                output_subdir = os.path.dirname(output_file_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                print(f"Processing: {input_file_path}")
                try:
                    extracted_content = extract_text(input_file_path)
                    if extracted_content.strip():  # Only write if there's content
                        with open(output_file_path, "w", encoding="utf-8") as f:
                            f.write(extracted_content)
                        print(f"  -> Saved: {output_file_path}")
                    else:
                        print(f"  -> Skipped (no content extracted): {input_file_path}")
                except Exception as e:
                    print(f"  -> Error processing {input_file_path}: {e}")
                    # Optionally, create an empty .txt file or a .txt file with the error message
                    try:
                        with open(
                            output_file_path + ".error.txt", "w", encoding="utf-8"
                        ) as f:
                            f.write(
                                f"Error processing original file {input_file_path}:\n{str(e)}"
                            )
                        print(
                            f"  -> Error details saved to: {output_file_path}.error.txt"
                        )
                    except Exception as ef:
                        print(
                            f"  -> Could not write error file for {input_file_path}: {ef}"
                        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF and DOCX files in a folder to TXT files."
    )
    parser.add_argument(
        "input_dir", help="The input directory containing .pdf and .docx files."
    )
    parser.add_argument(
        "output_dir", help="The output directory where .txt files will be saved."
    )

    args = parser.parse_args()

    # Check if extraction functions are usable (i.e., libraries were imported)
    if (
        "extract_text" not in globals()
        or globals()["extract_text"].__module__ == "__main__"
    ):  # Check if it's the dummy
        try:
            # Attempt a call that would fail if libs are missing, to trigger the ImportError from the dummy
            extract_text("dummy.pdf")
        except ImportError as e:
            # The error message is already printed by the try-except block at the top.
            # No need to print args.input_dir or args.output_dir here as they might not be used.
            print(
                "Script cannot proceed due to missing libraries. Please install them and try again."
            )
            return  # Exit if dependencies are not met

    process_files(args.input_dir, args.output_dir)
    print("\nConversion process finished.")


if __name__ == "__main__":
    main()
