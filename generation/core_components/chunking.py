# import glob
import glob
import hashlib
import json
import os
import io
from nltk.tokenize import sent_tokenize
import nltk  # NOTE to get this performing at all I need to make it so that chunking is cached and read from that cache instead ofbeing redone each time. Way too slow for large datasets. Which we are doing often and thus need to be cognizant of.

nltk.download("punkt_tab")
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    from PIL import Image
    from pdf2image import convert_from_path
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
        return "\n".join(full_text)

    def extract_text_from_pdf(path):
        """
        Extracts text from a copyable PDF using PyMuPDF.

        Args:
            path (str): The file path to the PDF.

        Returns:
            str: The extracted text.
        """
        text = ""
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text()
        except fitz.FileDataError as e:
            print(f"Warning: Skipping broken PDF file {path}. Error: {e}")
            return ""  # Return empty string for broken PDFs
        return text

    def extract_text_from_pdf_ocr(path):
        """
        Extracts text from a non-copyable PDF using OCR.

        Args:
            path (str): The file path to the PDF.

        Returns:
            str: The extracted text.
        """
        text = ""
        try:
            with fitz.open(path) as doc:
                for page_number, page in enumerate(doc):
                    # logger.info("Performing OCR on page %d", page_number + 1)
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes))
                    page_text = pytesseract.image_to_string(img)
                    text += page_text + "\n"
        except fitz.FileDataError as e:
            print(
                f"Warning: Skipping broken PDF during OCR for file {path}. Error: {e}"
            )
            return ""  # Return empty string for broken PDFs
        return text

    def remove_newlines_in_sentences(text):
        lines = text.split("\n")
        new_lines = []
        for line in lines:
            line = line.strip()
            if line:
                if line[-1] not in ".!?":
                    line += " "
                else:
                    line += "\n"
                new_lines.append(line)
        return "".join(new_lines)

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

        if ext == ".docx":
            # logger.info("Extracting text from DOCX file.")
            text = extract_text_from_docx(path)
        elif ext == ".pdf":
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
    print(
        "NOTE PDF and DOCX extraction will not work without the required libraries. Please install the required libraries to enable this functionality."
    )
    print("This is the error")
    print(e)

    def extract_text(file_path):
        raise ImportError(
            "PDF and DOCX extraction is not supported without the required libraries."
        )


# Add new try-except block for audio/video processing
try:
    import librosa
    import soundfile as sf
    from moviepy.editor import VideoFileClip
    from transformers import pipeline
    import tempfile
    import numpy as np

    def extract_text_from_audio_video(path):
        """
        Transcribes audio/video files using Whisper speech recognition.

        Args:
            path (str): Path to audio/video file

        Returns:
            str: Transcribed text
        """
        # Handle different file types
        _, ext = os.path.splitext(path)
        ext = ext.lower()

        if ext in [".mp3", ".wav", ".flac"]:
            # Load audio file directly
            audio, sr = librosa.load(path, sr=16000)
        elif ext in [".mp4", ".avi", ".mov"]:
            # Extract audio from video
            video = VideoFileClip(path)
            audio = video.audio
            # Save to temp file and reload with librosa
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                audio.write_audiofile(tmpfile.name, fps=16000)
                audio, sr = librosa.load(tmpfile.name, sr=16000)
                os.unlink(tmpfile.name)
        else:
            raise ValueError(f"Unsupported media format: {ext}")

        # Initialize Whisper pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            chunk_length_s=30,  # For handling long files
        )

        return pipe(audio.copy(), batch_size=8)["text"]

except ImportError as e:
    print(
        "NOTE: Audio/video processing requires librosa, soundfile, moviepy, and transformers. If you want to use videos as input for a pipeline, please install them. Otherwise, disregard this message."
    )

    # print("Error:", e)
    def extract_text_from_audio_video(path):
        raise ImportError(
            "Audio/video processing is not supported without required libraries"
        )


tokenizer = AutoTokenizer.from_pretrained("TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ")


def count_tokens(message):
    return len(tokenizer.encode(message))


def count_tokens_specific_model(model):
    tokenizer = AutoTokenizer.from_pretrained(model)

    def inner(message):
        return len(tokenizer.encode(message))

    return inner


def chunking_algorithm_file(
    file_path="./input/input.txt",
    max_token_length=1500,
    keep_folder_structure=False,
    input_dir="",
):
    """
    Combines format handling from Algorithm 1 with token-based chunking from Algorithm 2
    Adds minimum length filtering from Algorithm 1
    """
    chunks_with_source = []
    current_chunk = []
    token_count = 0

    # From Algorithm 1: Enhanced format handling
    if file_path.endswith(".pdf") or file_path.endswith(".docx"):
        content = extract_text(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()

    if not keep_folder_structure:
        basename = os.path.basename(file_path)
    else:
        specific_source = (
            file_path.replace(str(input_dir), "").lstrip("/").lstrip("./").lstrip("\\")
        )
        basename = specific_source
        # print("BASENAME")
        # print(basename)

    # basename = os.path.basename(file_path)

    # From Algorithm 2: Paragraph splitting and token-based logic
    paragraphs = content.split("\n\n")

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_token_count = count_tokens(paragraph)

        if paragraph_token_count > max_token_length:
            # Algorithm 2's sentence tokenization approach
            sentences = sent_tokenize(paragraph)
            for sentence in sentences:
                sentence_token_count = count_tokens(sentence)
                if token_count + sentence_token_count <= max_token_length:
                    current_chunk.append(sentence)
                    token_count += sentence_token_count
                else:
                    chunks_with_source.append(
                        {"text": " ".join(current_chunk), "metadata": basename}
                    )
                    current_chunk = [sentence]
                    token_count = sentence_token_count
        else:
            # Algorithm 2's paragraph accumulation logic
            if token_count + paragraph_token_count <= max_token_length:
                current_chunk.append(paragraph)
                token_count += paragraph_token_count
            else:
                chunks_with_source.append(
                    {"text": " ".join(current_chunk), "metadata": basename}
                )
                current_chunk = [paragraph]
                token_count = paragraph_token_count

    # Add final chunk
    if current_chunk:
        chunks_with_source.append(
            {"text": " ".join(current_chunk), "metadata": basename}
        )

    # From Algorithm 1: Minimum length filtering
    chunks_with_source = [
        chunk for chunk in chunks_with_source if len(chunk["text"]) >= 50
    ]

    return chunks_with_source


def chunking_algorithm_str(
    content=None,
    source_name="None",
    max_token_length=1500,
    keep_folder_structure=False,
    input_dir="",
):
    """
    Combines format handling from Algorithm 1 with token-based chunking from Algorithm 2
    Adds minimum length filtering from Algorithm 1
    """
    chunks_with_source = []
    current_chunk = []
    token_count = 0
    if not keep_folder_structure:
        basename = os.path.basename(source_name)
    else:
        specific_source = (
            source_name.replace(str(input_dir), "")
            .lstrip("/")
            .lstrip("./")
            .lstrip("\\")
        )
        basename = specific_source
        # print("BASENAME")
        # print(basename)

    # From Algorithm 2: Paragraph splitting and token-based logic
    paragraphs = content.split("\n\n")

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_token_count = count_tokens(paragraph)

        if paragraph_token_count > max_token_length:
            # Algorithm 2's sentence tokenization approach
            sentences = sent_tokenize(paragraph)
            for sentence in sentences:
                sentence_token_count = count_tokens(sentence)
                if token_count + sentence_token_count <= max_token_length:
                    current_chunk.append(sentence)
                    token_count += sentence_token_count
                else:
                    chunks_with_source.append(
                        {"text": " ".join(current_chunk), "metadata": basename}
                    )
                    current_chunk = [sentence]
                    token_count = sentence_token_count
        else:
            # Algorithm 2's paragraph accumulation logic
            if token_count + paragraph_token_count <= max_token_length:
                current_chunk.append(paragraph)
                token_count += paragraph_token_count
            else:
                chunks_with_source.append(
                    {"text": " ".join(current_chunk), "metadata": basename}
                )
                current_chunk = [paragraph]
                token_count = paragraph_token_count

    # Add final chunk
    if current_chunk:
        chunks_with_source.append(
            {"text": " ".join(current_chunk), "metadata": basename}
        )

    # From Algorithm 1: Minimum length filtering
    chunks_with_source = [
        chunk for chunk in chunks_with_source if len(chunk["text"]) >= 50
    ]

    return chunks_with_source


def read_jsonl_completions(input_dir="./input"):
    # reads the 'text' key of every object in each .json and .jsonl file in a directory.
    # So [{"text": "..."}, {"text": "....."}, ...] -> list of texts and source filenames

    source_files = []
    for ext in [".json", ".jsonl"]:
        path = f"{input_dir}/**/*{ext}"
        source_files += glob.glob(path, recursive=True)

    output_list = []
    for file_path in source_files:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".json"):
                data = json.load(f)  # Load entire JSON array
            else:  # JSONL
                data = [json.loads(line) for line in f]

            for item in data:
                if "text" in item:
                    output_list.append(
                        {"text": item["text"], "metadata": os.path.basename(file_path)}
                    )

    return output_list


def read_jsonl_file(file_path):
    """
    Reads a single JSONL file and extracts items with 'text' key.

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        list: List of dictionaries with 'text' and 'metadata' keys
    """
    output_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".json"):
                data = json.load(f)  # Load entire JSON array
            else:  # JSONL
                data = [json.loads(line) for line in f]

            for item in data:
                if "text" in item:
                    output_list.append(
                        {"text": item["text"], "metadata": os.path.basename(file_path)}
                    )
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return output_list


def read_text(
    input_dir=None,
    extensions=[".txt", ".md", ".pdf", ".docx", ".epub", ".html", ".jsonl"],
    output_dir=None,
):
    """
    Reads text files from a directory, handling various formats and optionally caching results.

    Args:
        input_dir (str): Directory to search for files.
        extensions (list): File extensions to include.
        output_dir (str, optional): Directory to save/load cached read results. If None, caching is disabled.

    Returns:
        list: List of dictionaries, each with 'text' and 'metadata' keys.
    """
    cache_path = None
    if output_dir:
        # Use absolute path for input_dir for consistent hashing
        abs_input_dir = os.path.abspath(input_dir)
        # Create a deterministic hash based on input directory and sorted extensions
        extensions_string = json.dumps(sorted(extensions))
        hash_content = f"{abs_input_dir}_{extensions_string}"
        hasher = hashlib.md5(hash_content.encode())
        cache_hash = hasher.hexdigest()
        cache_filename = f"read_cache_{cache_hash}.json"
        cache_path = os.path.join(output_dir, cache_filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Check if cache exists
        if os.path.exists(cache_path):
            print(f"Loading cached read results from {cache_path}")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: Read cache file {cache_path} is corrupted. Re-reading."
                )
            except Exception as e:
                print(
                    f"Warning: Could not load read cache file {cache_path} due to error: {e}. Re-reading."
                )

    # Proceed with reading if no cache or cache failed
    source_texts = []
    for extension in extensions:
        print("Checking extension")
        print(extension)
        path = f"{input_dir}/**/*" + extension
        source_texts = source_texts + glob.glob(path, recursive=True)

    if source_texts:
        print("Source texts found:")
        print(source_texts)
    else:
        print(f"No source texts found in: {input_dir}")
        # Print the full absolute path of the input directory
        absolute_input_dir = os.path.abspath(input_dir)
        print(f"Full absolute path of input directory: {absolute_input_dir}")

    output_text_list = []
    for text in tqdm(source_texts, desc="Reading"):
        # Confirm if the path points to a file, not a folder with an extension
        if os.path.isdir(text):
            print(f"Skipping {text} as it's a directory with an extension, not a file")
            continue
        if text.endswith(".pdf") or text.endswith(".docx"):
            loaded = extract_text(text)
        elif text.endswith(".jsonl"):
            loaded = read_jsonl_file(text)
            loaded_list = [
                {"text": item_obj["text"], "metadata": text} for item_obj in loaded
            ]
            output_text_list.extend(loaded_list)
            continue
        else:
            # Handle potential decoding errors by replacing problematic characters
            loaded = open(text, "r", encoding="utf-8", errors="replace").read()

        output_text_list.append(
            {"text": loaded, "metadata": text.replace(input_dir, "").lstrip("/")}
        )

    # Save to cache if output_dir was provided
    if cache_path:
        print(f"Saving read results to cache: {cache_path}")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(output_text_list, f, indent=4)
        except Exception as e:
            print(
                f"Warning: Could not save read cache file {cache_path} due to error: {e}"
            )

    return output_text_list


def write_text(output_dir, text_list):
    """
    Writes a list of text objects to files in the specified output directory.

    Args:
        output_dir (str): Directory where files will be written. Will be created if it doesn't exist.
        text_list (list): List of dictionaries with 'text' and 'metadata' keys, as returned by read_text.
                         The 'metadata' field should contain the relative file path.

    Returns:
        int: Number of files successfully written.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    files_written = 0

    for item in tqdm(text_list, desc="Writing files"):
        if "text" not in item or "metadata" not in item:
            print(f"Warning: Skipping item missing required fields: {item}")
            continue

        # Get the relative file path from metadata
        relative_path = item["metadata"]

        # Create the full output path
        full_path = os.path.join(output_dir, relative_path)

        # Create any necessary subdirectories
        file_dir = os.path.dirname(full_path)
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)

        try:
            # Write the text content to the file
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(item["text"])
            files_written += 1
        except Exception as e:
            print(f"Error writing file {full_path}: {e}")

    print(f"Successfully wrote {files_written} files to {output_dir}")
    return files_written


def chunk_text_list(
    text_list,
    chunk_size=1500,
    keep_folder_structure=False,
    input_dir="",
    output_dir=None,
):
    """
    Chunks a list of text dictionaries, with optional caching.

    Args:
        text_list (list): List of dictionaries, each with 'text' and 'metadata' keys.
        chunk_size (int): Maximum token length for chunks.
        keep_folder_structure (bool): Whether to preserve folder structure in metadata.
        input_dir (str): Input directory path (used if keep_folder_structure is True).
        output_dir (str, optional): Directory to save/load cached chunks. If None, caching is disabled.

    Returns:
        list: List of chunked dictionaries.
    """
    cache_path = None
    if output_dir:
        # Create a deterministic hash based on sorted metadata
        metadata_list = sorted([item["metadata"] for item in text_list])
        metadata_string = json.dumps(
            metadata_list
        )  # Use json dump for consistent string representation
        hasher = hashlib.md5(metadata_string.encode())
        cache_hash = hasher.hexdigest()
        cache_filename = f"chunk_cache_{cache_hash}_size{chunk_size}.json"
        cache_path = os.path.join(output_dir, cache_filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Check if cache exists
        if os.path.exists(cache_path):
            print(f"Loading cached chunks from {cache_path}")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Cache file {cache_path} is corrupted. Re-chunking.")
            except Exception as e:
                print(
                    f"Warning: Could not load cache file {cache_path} due to error: {e}. Re-chunking."
                )

    # Proceed with chunking if no cache or cache failed
    chunks = []
    for text in tqdm(text_list, desc="Chunking"):
        new_chunks = chunking_algorithm_str(
            text["text"],
            text["metadata"],
            max_token_length=chunk_size,
            keep_folder_structure=keep_folder_structure,
            input_dir=input_dir,
        )
        chunks.extend(new_chunks)

    # Save to cache if output_dir was provided
    if cache_path:
        print(f"Saving chunked data to cache: {cache_path}")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save cache file {cache_path} due to error: {e}")

    return chunks


def read_and_chunk_text(
    input_dir="./input",
    extensions=[".txt", ".md", ".pdf", ".docx", ".epub", ".html", ".jsonl"],
    chunk_size=1500,
    use_subset=False,
    subset_size=1500,
    keep_folder_structure=False,
    output_dir=None,
    seed=1048596,
):  # for splitting up documents
    # Print source texts for debugging
    # source_texts = []
    # for extension in extensions:
    #         path = f"{input_dir}/**/*" + extension
    #         source_texts = source_texts + glob.glob(path, recursive=True)

    # Use composition of read_text and chunk_text_list
    text_list = read_text(input_dir, extensions, output_dir=output_dir)
    sentence_chunks = chunk_text_list(
        text_list, chunk_size, keep_folder_structure, input_dir, output_dir=output_dir
    )
    # print("Ran this")

    if (
        use_subset
    ):  # NOTE that because we subset after chunking, we can reuse the same chunk cache even for different runs with different seeds! Automatically! Good design by accident
        if len(sentence_chunks) > subset_size:
            # Sort chunks by hash of their text for deterministic selection
            sentence_chunks = sorted(
                sentence_chunks,
                key=lambda x: hashlib.md5(
                    str(seed).encode() + x["text"].encode()
                ).hexdigest(),
            )[:subset_size]

    return sentence_chunks


def subset_text_list(text_list, subset_size=1500, seed=1048596):
    if len(text_list) > subset_size:
        # Sort chunks by hash of their text for deterministic selection
        text_list = sorted(
            text_list,
            key=lambda x: hashlib.md5(
                str(seed).encode() + x["text"].encode()
            ).hexdigest(),
        )[:subset_size]
    return text_list


def read_sharegpt_conversations(input_dir="./input"):
    # reads all .jsonl or .json files in the input
    # each has a list of messages with a "conversations" key
    # each message has a "from" and "value" key
    # from is either "human" or "gpt"
    # value is the message content

    # So the structure is like:
    # [
    #     {
    #         "conversations": [
    #             {"from": "human", "value": "Hello, how are you?"},
    #             {"from": "gpt", "value": "I'm good, thank you!"}
    #             {"from": "human", "value": "What is the capital of France?"},
    #             {"from": "gpt", "value": "The capital of France is Paris."}
    #             ...
    #         ]
    #     }
    # ]
    # OR if with a system prompt:
    # [
    #     {
    #         "conversations": [
    #             {"from": "system", "value": "You are a helpful assistant."},
    #             {"from": "human", "value": "Hello, how are you?"},
    #             {"from": "gpt", "value": "I'm good, thank you!"}
    #             ...
    #         ]
    # we want to return a list of items with a "conversations" key unchanged, and a "text" key which is the pretty stringification of the list of messages
    # stringification is something like:
    # for item in input_data["conversations"]:
    #         conv_history_str = conv_history_str + "\n" + item["from"].upper() + ":\n" + item["value"]
    # all files get concatenated together

    source_files = []
    for ext in [".json", ".jsonl"]:
        path = f"{input_dir}/**/*{ext}"
        source_files += glob.glob(path, recursive=True)

    output_list = []
    for file_path in source_files:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".json"):
                data = json.load(f)  # Load entire JSON array
            else:  # JSONL
                data = [json.loads(line) for line in f]

        for item in data:
            if "conversations" in item:
                # Create a stringified version of the conversation
                conv_history_str = ""
                for message in item["conversations"]:
                    conv_history_str += (
                        "\n" + message["from"].upper() + ":\n" + message["value"]
                    )

                # Create stringified version without first message
                conv_history_str_no_first = ""
                for message in item["conversations"][1:]:
                    conv_history_str_no_first += (
                        "\n" + message["from"].upper() + ":\n" + message["value"]
                    )

                # Hash the conversation without first message
                hash_obj = hashlib.md5(conv_history_str_no_first.strip().encode())
                hash_key = hash_obj.hexdigest()

                # Add to output list with both the original conversations and the stringified text
                output_list.append(
                    {
                        "conversations": item["conversations"],
                        "text": conv_history_str.strip(),
                        "hash_without_first": hash_key,
                    }
                )

    return output_list


def process_sharegpt_conversations(input_conversations):
    # the same as read except it takes in a list of conversations and does the operation, without reading from a file

    output_list = []
    for item in input_conversations:
        if "conversations" in item:
            # Create a stringified version of the conversation
            conv_history_str = ""
            for message in item["conversations"]:
                conv_history_str += (
                    "\n" + message["from"].upper() + ":\n" + message["value"]
                )

            # Create stringified version without first message
            conv_history_str_no_first = ""
            for message in item["conversations"][1:]:
                conv_history_str_no_first += (
                    "\n" + message["from"].upper() + ":\n" + message["value"]
                )

            # Hash the conversation without first message
            hash_obj = hashlib.md5(conv_history_str_no_first.strip().encode())
            hash_key = hash_obj.hexdigest()

            # Add to output list with both the original conversations and the stringified text
            output_list.append(
                {
                    "conversations": item["conversations"],
                    "text": conv_history_str.strip(),
                    "hash_without_first": hash_key,
                }
            )

    return output_list


# configs will literally (while co-located within the folder of the pipeline for organization, will literally just be passed into the function of a node as kwargs)


# Add new reading function
def read_audio_video(
    input_dir="./input", extensions=[".mp3", ".wav", ".mp4", ".avi", ".mov", ".flac"]
):
    """
    Reads audio/video files and returns transcribed text with metadata.

    Args:
        input_dir (str): Directory to search for files
        extensions (list): File extensions to include

    Returns:
        list: List of dictionaries with 'text' and 'metadata' keys
    """
    source_files = []
    for extension in extensions:
        path = f"{input_dir}/**/*{extension}"
        source_files += glob.glob(path, recursive=True)

    output_list = []
    for file_path in source_files:
        try:
            transcribed_text = extract_text_from_audio_video(file_path)
            output_list.append(
                {"text": transcribed_text, "metadata": os.path.basename(file_path)}
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return output_list


def read_all_text(input_dir="./input", chunk_size=1500, output_dir=None):
    # reads both .json, .jsonl, and the read_text extensions as well
    # no chunking -> This description is now inaccurate as it uses read_and_chunk_text which does chunking. Kept for reference.
    # TODO: Consider renaming or refactoring if the 'no chunking' aspect was important.
    list1 = read_jsonl_completions(input_dir)
    # Pass output_dir down to read_and_chunk_text
    list2 = read_and_chunk_text(
        input_dir=input_dir, chunk_size=chunk_size, output_dir=output_dir
    )
    return list1 + list2


def count_total_tokens(output_list, count_tokens_fn=None):
    """
    Counts the total number of tokens across all texts in the output list.

    Args:
        output_list (list): List of dictionaries with 'text' keys
        count_tokens_fn (callable, optional): Function to count tokens in text.
            If None, a simple word-based tokenization is used.

    Returns:
        int: Total token count across all texts
    """
    total_tokens = 0

    # Use simple word-based tokenization if no function provided
    if count_tokens_fn is None:

        def count_tokens_fn(text):
            return len(text.split())

    for item in output_list:
        if "text" in item and item["text"]:
            total_tokens += count_tokens_fn(item["text"])

    return total_tokens


def slice_conversation_history(conversation_item, pair_idx):
    """
    Slices the conversation history up to the specified pair index, excluding the system prompt.

    Args:
        conversation_item (dict): A dictionary containing a 'conversations' key
                                  with a list of message dictionaries.
        pair_idx (int): The 0-based index of the human-GPT pair *up to which*
                        the history should be retrieved (exclusive).

    Returns:
        list: A list of message dictionaries representing the history before
              the specified pair, or an empty list if input is invalid or
              pair_idx is 0.
    """
    if (
        not conversation_item
        or "conversations" not in conversation_item
        or not isinstance(conversation_item["conversations"], list)
    ):
        return []

    messages = conversation_item["conversations"]
    start_idx = 0
    if messages and messages[0]["from"] == "system":
        start_idx = 1  # Skip system prompt

    end_idx_exclusive = start_idx + (pair_idx * 2)

    if end_idx_exclusive < start_idx or end_idx_exclusive > len(messages):
        if pair_idx == 0:
            return []
        else:
            end_idx_exclusive = len(messages)

    return messages[start_idx:end_idx_exclusive]


def format_conversation_history(message_list):
    """
    Formats a list of message dictionaries into a string representation,
    truncating to the last 30000 tokens if necessary.

    Args:
        message_list (list): A list of message dictionaries, each with 'from' and 'value'.

    Returns:
        str: A formatted string of the conversation history, potentially truncated.
    """
    conv_history_str = ""
    for message in message_list:
        if "from" in message and "value" in message:
            # Add a space before the newline for potentially better tokenization later
            conv_history_str += f" \n{message['from'].upper()}:\n{message['value']}"

    # Calculate token count
    token_count = count_tokens(conv_history_str)

    # Truncate if necessary
    if token_count > 30000:
        # Encode the string to get tokens
        encoded_tokens = tokenizer.encode(conv_history_str)
        # Keep only the last 30000 tokens
        truncated_tokens = encoded_tokens[-30000:]
        # Decode back to string, skipping special tokens that might be added
        truncated_str = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        # Prepend the truncation indicator
        final_str = "...truncated..." + truncated_str
    else:
        final_str = conv_history_str

    return final_str.strip()


# Define a function that takes a conversation and the pair_idx, and slices the conversation up until that point (not including system prompt)
# and another function that formats it like the conv_history_str with the message from.upper:\nmsg["value"]
# and then we store a reference to the original conversation in the original dict instead of storing conv_history_str? Would that actually solve it?


def _extract_sharegpt_pairs(conversations, conv_idx=0, source_file=None):
    """
    Helper function to extract human-GPT pairs from ShareGPT conversations.

    Args:
        conversations (list): List of ShareGPT conversation items
        conv_idx (int): Index of this conversation
        source_file (str, optional): Source filename

    Returns:
        list: Extracted human-GPT pairs
    """
    output_pairs = []

    if "conversations" in conversations:
        # Check if there's a system prompt
        system_prompt = "not present"
        start_idx = 0

        if (
            conversations["conversations"]
            and conversations["conversations"][0]["from"] == "system"
        ):
            system_prompt = conversations["conversations"][0]["value"]
            start_idx = 1

        # Extract human-GPT pairs
        messages = conversations["conversations"][start_idx:]

        # Process pairs (human followed by GPT)
        for i in range(0, len(messages) - 1, 2):
            if (
                i + 1 < len(messages)
                and messages[i]["from"] == "human"
                and messages[i + 1]["from"] == "gpt"
            ):
                human_msg = messages[i]
                gpt_msg = messages[i + 1]

                # Create stringified text version of just this pair
                pair_text = (
                    f"HUMAN:\\n{human_msg['value']}\\n\\nGPT:\\n{gpt_msg['value']}"
                )

                # Calculate message indices in the original conversation
                human_idx = i + start_idx
                gpt_idx = i + 1 + start_idx

                # Add to output list
                pair_data = {
                    "human": human_msg["value"],
                    "gpt": gpt_msg["value"],
                    "system": system_prompt,
                    "text": pair_text,
                    "conv_idx": conv_idx,  # Which conversation in the original list
                    "pair_idx": i // 2,  # Which pair in the original conversation
                    "original_indices": [
                        human_idx,
                        gpt_idx,
                    ],  # Original message indices
                }

                if source_file:
                    pair_data["source_file"] = source_file

                output_pairs.append(pair_data)

    return output_pairs


def read_sharegpt_pairs(input_dir="./input"):
    """
    Reads all .jsonl or .json files containing ShareGPT conversations
    and returns a list with individual human-GPT message pairs.

    Args:
        input_dir (str): Directory containing ShareGPT conversation files

    Returns:
        list: List of dictionaries with human-GPT pairs and metadata
    """
    source_files = []
    for ext in [".json", ".jsonl"]:
        path = f"{input_dir}/**/*{ext}"
        source_files += glob.glob(path, recursive=True)

    output_pairs = []
    output_convs = (
        {}
    )  # list of conversations lined up with the pairs so that the indices match
    for file_path in source_files:
        output_convs[file_path] = []
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".json"):
                data = json.load(f)  # Load entire JSON array
            else:  # JSONL
                data = [json.loads(line) for line in f]

        # Track which conversation this came from
        for conv_idx, item in enumerate(data):
            pairs = _extract_sharegpt_pairs(item, conv_idx, file_path)
            output_pairs.extend(pairs)
            output_convs[file_path].append(item)

    return output_pairs, output_convs


def process_sharegpt_pairs(conversations_list):
    """
    Process a list of ShareGPT conversations and convert them to individual human-GPT pairs.

    Args:
        conversations_list (list): List of dictionaries containing ShareGPT conversations

    Returns:
        list: List of dictionaries with human-GPT pairs and metadata
    """
    output_pairs = []
    output_convs = {"default": []}

    for conv_idx, conversation in enumerate(conversations_list):
        pairs = _extract_sharegpt_pairs(conversation, conv_idx)
        source_file = "default"
        for pair in pairs:
            if not "source_file" in pair:
                pair["source_file"] = "default"
            else:
                source_file = pair["source_file"]
        output_pairs.extend(pairs)
        if not source_file in output_convs:
            output_convs[source_file] = []
        output_convs[source_file].append(conversation)

    return output_pairs, output_convs
