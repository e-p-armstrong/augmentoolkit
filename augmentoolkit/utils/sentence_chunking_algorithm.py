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
    source_name = file_path.replace(".txt", "")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    # try:
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         content = f.read()
    # except Exception as e:
    #     print(f"\nError reading file {file_path}: {e}\n")
    #     return []

    paragraphs = content.split(
        "\n\n"
    )  # Assuming paragraphs are separated by two newlines # TODO change so that if the length is 1 after this, split by tabs instead

    # HOW TO DO IT probably:
    # add tokens to the paragraph until we reach the max length,
    # create chunks out of the remainder of the paragraph (split at max chunk length until it's done)
    # if the final chunk does not have the max length, then make it the new current chunk, set the current token count to its length, and continue with the for loop.
    # Ensure max_char_length is an integer
    max_char_length = int(max_char_length)

    for paragraph in paragraphs:
        paragraph = paragraph.strip()  # Remove leading and trailing whitespace
        if not paragraph:  # Skip empty paragraphs
            continue

        paragraph_char_count = len(paragraph)

        # Check if the paragraph itself exceeds the max token length
        if paragraph_char_count > max_char_length:

            # Fallback to character chunking for this paragraph
            end_index = (
                max_char_length - char_count
            )  # after this we will take max_char_length chunks starting from end index until the end of the paragraph
            current_chunk.append(paragraph[:end_index])
            # characters = list(paragraph)
            chunks_with_source.append(
                {
                    "paragraph": "".join(current_chunk), 
                    "metadata": source_name
                })
            current_chunk = []
            while end_index < paragraph_char_count:
                current_chunk.append(paragraph[end_index : end_index + max_char_length])
                chunks_with_source.append({
                    "paragraph": "".join(current_chunk), 
                    "metadata": source_name
                })
                current_chunk = []
                end_index += max_char_length

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

    return chunks_with_source