import re


import io
import chardet
import os

try:
    from PIL import Image
    from pdf2image import convert_from_path
    import textract
    import pytesseract
    import fitz  # pymupdf
    import docx
    def extract_text_from_docx(path):
        """
        Extracts text from a DOCX file.

        Args:
            path (str): The file path to the DOCX file.

        Returns:
            str: The extracted text.
        """
        doc = docx.Document(path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)

    def read_doc_file(file_path):
        return textract.process(file_path).decode("utf-8")

    def extract_text_from_pdf(path):
        """
        Extracts text from a copyable PDF using PyMuPDF.

        Args:
            path (str): The file path to the PDF.

        Returns:
            str: The extracted text.
        """
        text = ''
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def extract_text_from_pdf_ocr(path):
        """
        Extracts text from a non-copyable PDF using OCR.

        Args:
            path (str): The file path to the PDF.

        Returns:
            str: The extracted text.
        """
        text = ''
        with fitz.open(path) as doc:
            for page_number, page in enumerate(doc):
                # logger.info("Performing OCR on page %d", page_number + 1)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                page_text = pytesseract.image_to_string(img)
                text += page_text + '\n'
        return text

    def remove_newlines_in_sentences(text):
        lines = text.split('\n')
        new_lines = []
        for line in lines:
            line = line.strip()
            if line:
                if line[-1] not in '.!?':
                    line += ' '
                else:
                    line += '\n'
                new_lines.append(line)
        return ''.join(new_lines)

    def extract_text(path):
        """
        Extracts formatted text from a PDF or DOCX file.

        Args:
            path (str): The file path to the PDF or DOCX file.

        Returns:
            str: The extracted text in markdown format.
        """
        # Check the file extension
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        
        if ext == '.docx':
            # logger.info("Extracting text from DOCX file.")
            text = extract_text_from_docx(path)
        elif ext == '.pdf':
            # logger.info("Extracting text from PDF file.")
            text = extract_text_from_pdf(path)
            # logger.info("Extracted text length: %d", len(text))
            # If extracted text is too short, use OCR
            if len(text.strip()) < 100:
                # logger.info("Extracted text is too short, switching to OCR.")
                text = extract_text_from_pdf_ocr(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Remove newlines within sentences
        text = remove_newlines_in_sentences(text)
        
        return text
except ImportError as e:
    print("NOTE PDF and DOCX extraction will not work without the required libraries. Please install the required libraries to enable this functionality.")
    print("This is the error")
    print(e)
    def extract_text(file_path):
        raise Exception("PDF and DOCX extraction is not supported without the required libraries.")




def sentence_chunking_algorithm(file_path, max_char_length=1900):
    """
    This function takes a plaintext file and chunks it into paragraphs or sentences if the paragraph exceeds max_char_length.

    :param file_path: Path to the plaintext file
    :param max_char_length: The maximum char5acter length for a chunk
    :return: List of chunks with source text information
    """
    chunks_with_source = []
    current_chunk = []
    char_count = 0
    source_name = re.sub(r"\..*$", "", os.path.basename(file_path))
    
    if file_path.endswith(".pdf") or file_path.endswith(".docx"):
        content = extract_text(file_path)
    else:
        # with open(file_path, 'rb') as raw_file:
        #     raw_data = raw_file.read()
        #     result = chardet.detect(raw_data)
        #     file_encoding = result['encoding']
            
        # Now read the file with the detected encoding
        with open(file_path, "r", errors="ignore") as file:
            content = file.read()
        # with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        #     content = f.read()
    # try:
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         content = f.read()
    # except Exception as e:
    #     print(f"\nError reading file {file_path}: {e}\n")
    #     return []

    paragraphs = content.split(
        "\n\n"
    )
    max_char_length = int(max_char_length)

    for paragraph in paragraphs:
        paragraph = paragraph.strip()  # Remove leading and trailing whitespace
        if not paragraph:  # Skip empty paragraphs
            continue

        paragraph_char_count = len(paragraph)

        # Check if the paragraph itself exceeds the max token length
        if paragraph_char_count > max_char_length:
        # Fallback to character chunking for this paragraph
            end_index = 0
            
            while end_index < paragraph_char_count:
                chunk_end = min(end_index + max_char_length, paragraph_char_count)
                
                # Take until the next sentence ends (or we reach max_char_length*1.5)
                while (chunk_end < paragraph_char_count and 
                    paragraph[chunk_end] not in [".", "!", "?", "\n"] and 
                    chunk_end < end_index + max_char_length * 1.5):
                    chunk_end += 1
                # add one to chunk_end to include the punctuation IF it's not the last character
                if chunk_end < paragraph_char_count:
                    chunk_end += 1
                
                chunks_with_source.append({
                    "paragraph": paragraph[end_index:chunk_end],
                    "metadata": source_name
                })
                
                end_index = chunk_end

            # # handle the remainder of the paragraph
            # end_index = end_index - max_char_length
            # current_chunk.append(paragraph[end_index:])

            # char_count = paragraph_char_count - end_index
        else:
            if char_count + paragraph_char_count <= max_char_length:
                current_chunk.append(paragraph)
                char_count += paragraph_char_count
            else:
                chunks_with_source.append({
                    "paragraph": "".join(current_chunk), 
                    "metadata": source_name
                })
                current_chunk = [paragraph]
                char_count = paragraph_char_count

    # Add the last chunk if it exists
    if current_chunk:
        chunks_with_source.append({
                    "paragraph": "".join(current_chunk), 
                    "metadata": source_name
                })

    # filter out chunks with fewer than 50 characters
    chunks_with_source = [chunk for chunk in chunks_with_source if len(chunk["paragraph"]) >= 50]

    return chunks_with_source, content