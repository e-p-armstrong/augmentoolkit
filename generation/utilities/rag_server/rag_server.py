# TODO this is copy-pasted and redacted from another project. We need to hack it into a form that basically uses the same chat loop as llm_server.py (also this should be run through a main function that kicks off uvicorn, and have its args passed in via config) but this code will use RAG with the algorithm shown here (EXACTLY as shown) to enhance the context that the AI has for each message.
# Basically, use the structure of llm_server.py and use the spare parts of this to build a RAG chat loop with the same /generate route interface as llm_server.py.
# This thing's server function will take as an additional argument the path to an output dir. It will use the rag_source_data/rag_data_combined.jsonl as the source for the questions jsonl. Structure is the same as this script currently expects.
# It will also take as an additional argument the path to some input dir. That will be from where we draw the documents.
# Assume all libraries are installed.

import copy
import pickle
import platform
import time
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sys import argv
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, conlist, constr, StringConstraints, Field
import os
import asyncio
from nltk.tokenize import sent_tokenize
from typing import Annotated, List, Dict
import logging
import chromadb
from chromadb.utils import embedding_functions
import textwrap
import traceback

import uvicorn
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from tqdm import tqdm  # Import tqdm for progress bars
import secrets
from transformers import AutoTokenizer
import string
import re
import json
from generation.core_components.chunking import (
    count_tokens_specific_model,
    chunking_algorithm_str,
)
from generation.core_components.simple_chat_loop import (
    format_messages_into_string,
    get_assistant_prefix,
    get_stop_tokens,
)
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import subprocess
import shutil

from redis_config import set_progress  # For rmtree

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]


try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)


def find_robust_offsets(haystack_raw: str, needle_raw: str):
    if not needle_raw or not haystack_raw:
        return -1, -1
    needle_condensed = "".join(
        c for c in needle_raw if not c.isspace() and c not in string.punctuation
    )
    if not needle_condensed:
        return -1, -1
    haystack_condensed_chars = []
    map_condensed_idx_to_original_idx = []
    for original_idx, char in enumerate(haystack_raw):
        if not char.isspace() and char not in string.punctuation:
            haystack_condensed_chars.append(char)
            map_condensed_idx_to_original_idx.append(original_idx)
    haystack_condensed = "".join(haystack_condensed_chars)
    match_start_in_condensed = haystack_condensed.find(needle_condensed)
    if match_start_in_condensed == -1:
        # logger.debug(f"Robust offset search: Needle (condensed) not found in haystack (condensed).")
        # logger.debug(f"Needle (raw): '{needle_raw[:200]}...'")
        # logger.debug(f"Needle (condensed): '{needle_condensed[:200]}...'")
        # logger.debug(f"Haystack (raw length): {len(haystack_raw)}")
        # logger.debug(f"Haystack (condensed length): {len(haystack_condensed)}")
        return -1, -1
    original_start_offset = map_condensed_idx_to_original_idx[match_start_in_condensed]
    last_char_condensed_match_idx = match_start_in_condensed + len(needle_condensed) - 1
    if last_char_condensed_match_idx >= len(map_condensed_idx_to_original_idx):
        print(
            "Robust matching inconsistency: last_char_condensed_match_idx out of bounds."
        )
        return -1, -1
    original_end_offset_inclusive = map_condensed_idx_to_original_idx[
        last_char_condensed_match_idx
    ]
    original_end_offset = original_end_offset_inclusive + 1
    return original_start_offset, original_end_offset


def read_documents_file(documents_dir, source_metadata_path_relative):
    full_source_doc_path = os.path.join(documents_dir, source_metadata_path_relative)
    if not source_metadata_path_relative.endswith(
        ".txt"
    ):  # Ensure .txt extension if missing
        if not os.path.exists(full_source_doc_path) and os.path.exists(
            full_source_doc_path + ".txt"
        ):
            full_source_doc_path += ".txt"
        elif not os.path.exists(full_source_doc_path) and os.path.exists(
            os.path.splitext(full_source_doc_path)[0] + ".txt"
        ):  # handle if path has wrong ext
            full_source_doc_path = os.path.splitext(full_source_doc_path)[0] + ".txt"

    try:
        with open(full_source_doc_path, "r", encoding="utf-8") as f_doc:
            document_full_content = f_doc.read()
    except FileNotFoundError:
        print(f"FATAL: Source document file not found: {full_source_doc_path}")
        raise
    except Exception as e:
        print(f"FATAL: Error reading source document {full_source_doc_path}: {e}")
        raise
    return document_full_content


def vectorize_documents(
    bm25_cache_dir,
    bm25_corpus_data_path,
    bm25_index_path,
    collection_name,
    questions_jsonl_path,
    documents_dir,
    question_chunk_size,
    cache_dir,
):
    global chroma_client, collection, bm25_index, bm25_corpus_data

    os.makedirs(bm25_cache_dir, exist_ok=True)

    # Attempt to load BM25 data from cache first
    loaded_bm25_from_cache = False
    if os.path.exists(bm25_index_path) and os.path.exists(bm25_corpus_data_path):
        try:
            logger.info(f"Attempting to load BM25 index from {bm25_index_path}")
            with open(bm25_index_path, "rb") as f_idx:
                bm25_index = pickle.load(f_idx)
            logger.info(
                f"Attempting to load BM25 corpus data from {bm25_corpus_data_path}"
            )
            with open(bm25_corpus_data_path, "r", encoding="utf-8") as f_corpus:
                bm25_corpus_data = json.load(f_corpus)
            if bm25_index and bm25_corpus_data:
                logger.info(
                    "Successfully loaded BM25 index and corpus data from cache."
                )
                loaded_bm25_from_cache = True
            else:  # Should not happen if files exist and load without error
                bm25_index = None
                bm25_corpus_data = []
                logger.warning(
                    "BM25 cache files found but loading resulted in empty data. Will rebuild."
                )
        except Exception as e:
            logger.warning(
                f"Could not load BM25 from cache: {e}. Will rebuild if necessary."
            )
            bm25_index = None  # Ensure reset on error
            bm25_corpus_data = []

    chroma_client = chromadb.PersistentClient(path=os.path.join(cache_dir, "chroma_db"))
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    try:
        collection = chroma_client.get_collection(
            name=collection_name, embedding_function=embedding_function
        )
        logger.info(
            f"Found existing ChromaDB collection: {collection_name} with {collection.count()} items."
        )
    except Exception:
        logger.info(
            f"ChromaDB collection '{collection_name}' not found. Creating new collection."
        )
        collection = chroma_client.create_collection(
            name=collection_name, embedding_function=embedding_function
        )

    # If Chroma collection is empty, or if BM25 was not loaded from cache (implying potential desync or first run)
    # then, we (re)build everything from JSONL.
    if collection.count() == 0 or not loaded_bm25_from_cache:
        if collection.count() == 0:
            logger.info(
                f"Chroma collection '{collection_name}' is empty. Populating from: {questions_jsonl_path}"
            )
        if not loaded_bm25_from_cache:
            logger.info(
                f"BM25 data not loaded from cache. Will build/rebuild from: {questions_jsonl_path}"
            )
            # Reset BM25 data if we are rebuilding
            bm25_corpus_data = []
            bm25_index = None

        docs_to_embed = []
        metadatas_for_embedding = []
        ids_for_embedding = []
        embedding_item_id_counter = 0

        # Ensure bm25_corpus_data is reset if we are here because collection.count() == 0
        # and loaded_bm25_from_cache was true (which would be an inconsistent state we want to fix)
        if collection.count() == 0:
            bm25_corpus_data = []  # Explicitly reset for full rebuild
            bm25_index = None

        batch_size = 300
        jsonl_entries = []
        try:
            with open(questions_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    jsonl_entries.append(json.loads(line))
            logger.info(
                f"Successfully loaded {len(jsonl_entries)} entries from {questions_jsonl_path}"
            )
        except FileNotFoundError:
            print(f"FATAL: Questions JSONL file not found at {questions_jsonl_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"FATAL: Error decoding JSON from {questions_jsonl_path}: {e}")
            raise

        with tqdm(
            total=len(jsonl_entries), desc="Processing JSONL for Vector & BM25"
        ) as entry_pbar:
            for json_line_idx, item in enumerate(jsonl_entries):
                try:
                    question_text = item["question"]
                    source_text_content = item["source_text"]
                    source_metadata_path_relative = item["source_metadata"]
                    related_chunks_list = item["related_chunks"]

                    document_full_content = read_documents_file(
                        documents_dir, source_metadata_path_relative
                    )
                    current_passages_info = []
                    start_offset, end_offset = find_robust_offsets(
                        document_full_content, source_text_content
                    )

                    if start_offset == -1:
                        print(
                            f"source_text not found for JSONL entry {json_line_idx}. Q: '{question_text[:50]}...' Source: '{source_metadata_path_relative}'"
                        )
                        # print(f"Source text to find: '''{source_text_content[:200]}'''")
                        # print(f"Source text to find (full, repr): {repr(source_text_content)}")
                        # print(f"Document content (raw, repr): {repr(document_full_content)}")
                        # print(f"Length of source_text_content: {len(source_text_content)}")
                        # print(f"Length of document_full_content (raw): {len(document_full_content)}")
                        raise ValueError(
                            f"source_text not found for JSONL entry {json_line_idx}"
                        )
                    current_passages_info.append(
                        {
                            "text": source_text_content,
                            "metadata": source_metadata_path_relative,
                            "start_offset": start_offset,
                            "end_offset": end_offset,
                        }
                    )

                    for rel_chunk_idx, rel_chunk in enumerate(related_chunks_list):
                        chunk_text = rel_chunk["text"]
                        chunk_rel_path = rel_chunk["metadata"]
                        rel_doc_content = read_documents_file(
                            documents_dir, chunk_rel_path
                        )
                        start_offset_rel, end_offset_rel = find_robust_offsets(
                            rel_doc_content, chunk_text
                        )

                        if start_offset_rel == -1:
                            print(
                                f"FATAL: related_chunk text (idx {rel_chunk_idx}) not found in {chunk_rel_path}. Q: '{question_text[:50]}...' (JSONL idx: {json_line_idx}). Chunk: '{chunk_text[:50]}...'"
                            )
                            # print(f"Related chunk text to find (first 200 chars): '''{chunk_text}'''")
                            # print(f"Related chunk text to find (full, repr): {repr(chunk_text)}")
                            # print(f"Document content (raw, repr): {repr(rel_doc_content)}")
                            raise ValueError(
                                f"related_chunk text not found for JSONL entry {json_line_idx}, chunk {rel_chunk_idx}"
                            )
                        current_passages_info.append(
                            {
                                "text": chunk_text,
                                "metadata": chunk_rel_path,
                                "start_offset": start_offset_rel,
                                "end_offset": end_offset_rel,
                            }
                        )

                    relevant_passages_json_str = json.dumps(current_passages_info)

                    bm25_corpus_data.append(
                        {
                            "id": f"q_{json_line_idx}",
                            "original_question": question_text,
                            "relevant_passages_json": relevant_passages_json_str,
                            "source_file_path": source_metadata_path_relative,
                        }
                    )

                    # Using the imported chunking_algorithm_string as per original script's usage here
                    question_text_chunks = (
                        chunking_algorithm_str(  # This is the imported one
                            question_text,
                            max_token_length=question_chunk_size,
                            source_name=f"q_entry_{json_line_idx}",
                        )
                    )

                    for part_idx, q_chunk in enumerate(question_text_chunks):
                        if q_chunk["text"].strip():
                            docs_to_embed.append(q_chunk["text"])
                            metadatas_for_embedding.append(
                                {
                                    "original_question": question_text,
                                    "source_file_path": source_metadata_path_relative,
                                    "relevant_passages_json": relevant_passages_json_str,
                                }
                            )
                            ids_for_embedding.append(
                                f"q_{json_line_idx}_part_{part_idx}"
                            )
                            embedding_item_id_counter += 1

                            if len(docs_to_embed) >= batch_size:
                                collection.add(
                                    documents=docs_to_embed,
                                    metadatas=metadatas_for_embedding,
                                    ids=ids_for_embedding,
                                )
                                logger.info(
                                    f"Added batch of {len(docs_to_embed)} question chunks to Chroma."
                                )
                                docs_to_embed = []
                                metadatas_for_embedding = []
                                ids_for_embedding = []
                except Exception as e:
                    print(
                        f"FATAL: Error processing JSONL entry index {json_line_idx}, Q: '{item.get('question', 'N/A')[:50]}...'. Error: {str(e)}"
                    )
                    if not isinstance(
                        e, (FileNotFoundError, ValueError, json.JSONDecodeError)
                    ):
                        print(traceback.format_exc())
                    raise
                entry_pbar.update(1)
                entry_pbar.set_postfix(embedded_chunks=embedding_item_id_counter)

        if docs_to_embed:  # Add any remaining documents
            collection.add(
                documents=docs_to_embed,
                metadatas=metadatas_for_embedding,
                ids=ids_for_embedding,
            )
            logger.info(
                f"Added final batch of {len(docs_to_embed)} question chunks to Chroma."
            )

        logger.info(
            f"Completed Chroma embedding: {embedding_item_id_counter} chunks from {len(jsonl_entries)} JSONL entries."
        )

        if bm25_corpus_data:
            logger.info(
                f"Building BM25 index from {len(bm25_corpus_data)} original questions..."
            )
            tokenized_corpus_for_bm25 = [
                word_tokenize(doc["original_question"].lower())
                for doc in tqdm(bm25_corpus_data, desc="Tokenizing for BM25")
            ]
            bm25_index = BM25Okapi(tokenized_corpus_for_bm25)
            logger.info("BM25 index built successfully.")
            try:
                with open(bm25_index_path, "wb") as f_idx:
                    pickle.dump(bm25_index, f_idx)
                with open(bm25_corpus_data_path, "w", encoding="utf-8") as f_corpus:
                    json.dump(bm25_corpus_data, f_corpus, indent=2)
                logger.info(
                    f"BM25 index and corpus data saved to cache: {bm25_cache_dir}"
                )
            except Exception as e:
                print(f"Error saving BM25 data to cache: {e}")
        else:
            logger.warning(
                "No data available to build BM25 index from JSONL processing."
            )

    else:  # Chroma collection was populated and BM25 was successfully loaded from cache
        logger.info(
            f"ChromaDB collection '{collection_name}' ({collection.count()} items) and BM25 index ({len(bm25_corpus_data)} docs) are already loaded and synchronized."
        )

    # Final check: if Chroma has data but BM25 is still not available (e.g. cache deleted manually, JSONL not processed)
    if collection.count() > 0 and (not bm25_index or not bm25_corpus_data):
        logger.warning(
            "ChromaDB has data, but BM25 index or corpus is missing. "
            "BM25 search will be unavailable. Consider re-vectorizing or checking BM25 cache / JSONL path."
        )

    return chroma_client, collection


def reciprocal_rank_fusion(ranked_lists_of_ids, k=60):
    rrf_scores = {}
    for ranked_list in ranked_lists_of_ids:
        for rank, doc_id in enumerate(ranked_list):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)


def retrieve_relevant_chunks(collection_obj, query_text: str, top_k):
    global bm25_index, bm25_corpus_data

    if not collection_obj:
        logger.warning("Chroma collection is not initialized. Cannot retrieve chunks.")
        return []

    doc_metadata_map = {}

    vector_ranked_original_questions = []
    try:
        num_vector_candidates = top_k * 2
        chroma_results = collection_obj.query(
            query_texts=[query_text],
            n_results=num_vector_candidates,
            include=["metadatas", "documents"],
        )
        if chroma_results and chroma_results["ids"] and chroma_results["ids"][0]:
            for i in range(len(chroma_results["ids"][0])):
                meta = chroma_results["metadatas"][0][i]
                original_question = meta["original_question"]
                if original_question not in doc_metadata_map:
                    doc_metadata_map[original_question] = {
                        "relevant_passages_json": meta["relevant_passages_json"],
                        "source_file_path": meta["source_file_path"],
                    }
                if original_question not in vector_ranked_original_questions:
                    vector_ranked_original_questions.append(original_question)
        logger.debug(
            f"Vector search retrieved {len(vector_ranked_original_questions)} unique original questions for RRF."
        )
    except Exception as e:
        print(f"Error during ChromaDB query: {e}")
        vector_ranked_original_questions = []

    bm25_ranked_original_questions = []
    if bm25_index and bm25_corpus_data:
        try:
            tokenized_query = word_tokenize(query_text.lower())
            bm25_scores = bm25_index.get_scores(tokenized_query)
            scored_corpus_indices = [
                (score, i) for i, score in enumerate(bm25_scores) if score > 0
            ]
            scored_corpus_indices.sort(key=lambda x: x[0], reverse=True)

            num_bm25_candidates = top_k * 3
            for score, corpus_idx in scored_corpus_indices[:num_bm25_candidates]:
                corpus_item = bm25_corpus_data[corpus_idx]
                original_question = corpus_item["original_question"]
                if original_question not in doc_metadata_map:
                    doc_metadata_map[original_question] = {
                        "relevant_passages_json": corpus_item["relevant_passages_json"],
                        "source_file_path": corpus_item["source_file_path"],
                    }
                if original_question not in bm25_ranked_original_questions:
                    bm25_ranked_original_questions.append(original_question)
            logger.debug(
                f"BM25 search produced {len(bm25_ranked_original_questions)} unique candidate original questions for RRF."
            )
        except Exception as e:
            print(f"Error during BM25 search: {e}")
            bm25_ranked_original_questions = []
    else:
        logger.warning("BM25 index or corpus not available, skipping BM25 search.")
        # raise Exception("BM25 NOT AVAILABLE") # Original line, commented out to allow graceful degradation

    ranked_lists_for_rrf = []
    if vector_ranked_original_questions:
        ranked_lists_for_rrf.append(vector_ranked_original_questions)
    if bm25_ranked_original_questions:
        ranked_lists_for_rrf.append(bm25_ranked_original_questions)

    fused_results = []
    if ranked_lists_for_rrf:
        fused_results = reciprocal_rank_fusion(ranked_lists_for_rrf, k=60)
        logger.debug(f"RRF produced {len(fused_results)} fused results.")
    else:
        logger.debug("No results from vector or BM25 search to fuse.")
        return []

    retrieved_passages = []
    for original_question_text, rrf_score in fused_results[:top_k]:
        if original_question_text in doc_metadata_map:
            metadata_for_question = doc_metadata_map[original_question_text]
            try:
                passages_info_list = json.loads(
                    metadata_for_question["relevant_passages_json"]
                )
                for passage_data in passages_info_list:
                    retrieved_passages.append(
                        {
                            "text": passage_data["text"],
                            "source_file_path": passage_data["metadata"],
                            "original_question_trigger": original_question_text,
                            "char_offsets": (
                                passage_data["start_offset"],
                                passage_data["end_offset"],
                            ),
                        }
                    )
            except json.JSONDecodeError as e:
                print(
                    f"Error decoding relevant_passages_json for Q: '{original_question_text[:50]}...'. Error: {e}"
                )
        else:
            logger.warning(
                f"Original question '{original_question_text[:50]}...' from RRF not in doc_metadata_map."
            )

    final_passages = []
    seen_passages_keys = set()
    for passage in retrieved_passages:
        passage_key = (
            passage["source_file_path"],
            passage["char_offsets"][0],
            passage["char_offsets"][1],
            passage["text"],
        )
        if passage_key not in seen_passages_keys:
            final_passages.append(passage)
            seen_passages_keys.add(passage_key)

    logger.info(
        f"Retrieved {len(final_passages)} unique passages after RRF for query: '{query_text[:50]}...'"
    )
    return final_passages


def stringify_rag_chunks(input_chunks):
    result_string = ""
    if not input_chunks:
        return "No relevant contextual information was found in the documents for the query.\n"
    for idx, chunk_info in enumerate(input_chunks):
        if (
            not chunk_info["text"] or not chunk_info["text"].strip()
        ):  # Skip empty chunks
            continue
        result_string += f"""Chunk {idx}
Source: {chunk_info["source_file_path"]} (Characters: {chunk_info["char_offsets"][0]}-{chunk_info["char_offsets"][1]})
(Retrieved based on question: "{chunk_info["original_question_trigger"]}")
---
{chunk_info["text"]}
---

"""
    if not result_string:  # If all chunks were empty
        return "No relevant contextual information was found in the documents for the query (after filtering empty chunks).\n"
    return result_string


# TODO I will use the current thing instead, from openai server
# def convert_messages_to_prompt(messages, prefix=None, chat_mode=False):
#     message_dicts = [m.model_dump() if hasattr(m, 'model_dump') else m for m in messages]
#     # logger.debug(f"Converting messages to prompt with chat_mode: {chat_mode}")
#     # logger.debug(f"Messages: {message_dicts}")
#     result = ""
#     if chat_mode == "ooba":
#         for message in message_dicts: result += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
#         result += "<|im_start|>assistant\n"
#         if prefix: result += prefix
#     elif chat_mode == "chatml":
#         result = f"<|im_start|>user\n{message_dicts[0]['content']}"
#         for message in message_dicts[1:]:
#             curr_prefix = prefix if message['role'] == 'assistant' else "You:"
#             result += f"\n{curr_prefix} {message['content'].strip()}"
#         result += "\n<|im_end|>\n<|im_start|>assistant\n"
#         if prefix: result += prefix
#     # ... (other chat modes from your original script) ...
#     elif chat_mode == "recite": # This is the one used in /generate
#         result = ""
#         for idx, message in enumerate(message_dicts):
#             role = message['role']
#             content = message['content'].strip() # Ensure content is stripped
#             if role == 'system':
#                 result += f"Instruction: {content} **Finished.**\n"
#             elif role == 'user':
#                 result += f"Human: {content} **Finished.**\n"
#             elif role == 'assistant':
#                 result += f"AI: {content} **Finished.**\n"
#         result += "AI: Thought Process:" # No prefix here as per original
#     else: # Fallback or other modes
#         # Using a simple concatenation for brevity, expand if other modes are critical
#         # For now, ensure it handles the 'recite' mode correctly as it's used.
#         # The original script had many modes, ensure the used one is correct.
#         # The 'recite' mode was explicitly used in /generate
#         # Defaulting to a simple join if mode not handled, or raise error
#         logger.warning(f"Unsupported or simplified chat_mode: {chat_mode} in convert_messages_to_prompt. Using basic join or 'recite' logic.")
#         # Re-implementing 'recite' here as it's the primary one used.
#         result = ""
#         for idx, message in enumerate(message_dicts):
#             role = message['role']
#             content = message['content'].strip()
#             if role == 'system': result += f"Instruction: {content} **Finished.**\n"
#             elif role == 'user': result += f"Human: {content} **Finished.**\n"
#             elif role == 'assistant': result += f"AI: {content} **Finished.**\n"
#         result += "AI: Thought Process:"
#         if prefix and chat_mode != "recite": # Prefix for recite is handled by Thought Process
#              result += prefix

#     # logger.debug(f"Generated prompt: {result[:500]}...") # Log only a part of long prompts
#     return result


async def rag_server(
    prompt_path,
    template_path,
    gguf_model_path,
    context_length,
    documents_dir: str,
    questions_jsonl_path: str,
    question_chunk_size: int,
    top_k: int,
    cache_dir: str,
    max_shrink_iterations,
    collection_name: str = "questions_collection",
    llama_path="./llama.cpp",  # customizable llama.cpp path
    port=8003,
    task_id=None,
    **kwargs,
):

    with open(prompt_path, "r") as f:
        prompt = f.read()

    with open(template_path, "r") as f:
        template = f.read()

    # Get parent directory of gguf model path
    model_dir = os.path.dirname(gguf_model_path)

    # Initialize model-specific token counting
    try:
        count_tokens = count_tokens_specific_model(
            model_dir
        )  # Assumes that tokenizer and sucha re still inside the same dir as the saved gguf model.
    except Exception as e:
        print(e)
        print(
            "\n\nYou probably deleted the tokenizer and other ssuch things from the model directory after you got your quantized model. However, the model tokenizer is used to count tokens before sending it off to llama.cpp, so you shouldn't have done that. You can delete model files (*.safetensors) to save space, but leave the tokenizer alone."
        )
        print(
            "To fix this, re-run the datagen pipeline that produced this model. It won't re-train, but it will re-download"
        )
        raise

    # First thing's first, ragify the docs

    app = FastAPI(
        title="Augmentoolkit Custom Model RAG-Enabled API Server", version="0.1.0"
    )

    chroma_client = None
    collection = None
    bm25_index = None
    bm25_corpus_data = []

    # BM25 Persistence paths
    BM25_CACHE_DIR = os.path.join(cache_dir, "bm25_cache")
    BM25_INDEX_PATH = os.path.join(BM25_CACHE_DIR, "bm25_index.pkl")
    BM25_CORPUS_DATA_PATH = os.path.join(BM25_CACHE_DIR, "bm25_corpus_data.json")

    logger.info("Application startup: Initializing RAG system (ChromaDB & BM25)...")
    chroma_client, collection = vectorize_documents(
        bm25_cache_dir=BM25_CACHE_DIR,
        bm25_corpus_data_path=BM25_CORPUS_DATA_PATH,
        bm25_index_path=BM25_INDEX_PATH,
        documents_dir=documents_dir,
        questions_jsonl_path=questions_jsonl_path,
        collection_name=collection_name,
        question_chunk_size=question_chunk_size,
        cache_dir=cache_dir,
    )
    logger.info("RAG system (ChromaDB & BM25) initialization complete.")

    # Initialize model-specific token counting
    try:
        count_tokens = count_tokens_specific_model(
            model_dir
        )  # Assumes that tokenizer and sucha re still inside the same dir as the saved gguf model.
    except Exception as e:
        print(e)
        print(
            "\n\nYou probably deleted the tokenizer and other ssuch things from the model directory after you got your quantized model. However, the model tokenizer is used to count tokens before sending it off to llama.cpp, so you shouldn't have done that. You can delete model files (*.safetensors) to save space, but leave the tokenizer alone."
        )
        print(
            "To fix this, re-run the datagen pipeline that produced this model. It won't re-train, but it will re-download"
        )
        raise

    # llama.cpp
    if not os.path.exists(llama_path):
        print("llama.cpp directory not found. Cloning repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggml-org/llama.cpp.git"], check=True
        )
        subprocess.run(
            ["git", "checkout", "b775345d788ac16260e7eef49e11fe57ee5677f7"],
            cwd="llama.cpp",
            check=True
        )

        # Check if llama-server exists
        llama_server_path = os.path.join(llama_path, "build", "bin", "llama-server")
        if platform.system() == "Windows":
            llama_server_path += ".exe"

        if not os.path.exists(llama_server_path):
            print("llama-server not found. Building llama.cpp...")

            # Detect if NVIDIA GPU is available
            has_nvidia_gpu = False
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=False, text=True)
                has_nvidia_gpu = result.returncode == 0
            except FileNotFoundError:
                has_nvidia_gpu = False

            # Build with appropriate flags
            build_cmd = ["cmake", "-B", "build"]
            if has_nvidia_gpu:
                build_cmd.append("-DGGML_CUDA=ON")
                print("NVIDIA GPU detected. Building with CUDA support...")
            else:
                print("No NVIDIA GPU detected. Building CPU-only version...")

            # Run cmake configure
            subprocess.run(build_cmd, cwd=llama_path, check=True)

            # Build the project
            subprocess.run(
                ["cmake", "--build", "build", "--config", "Release"],
                cwd=llama_path,
                check=True,
            )

    # Build llama-server path
    llama_server_path = os.path.join(llama_path, "build", "bin", "llama-server")
    if platform.system() == "Windows":
        llama_server_path += ".exe"

    # Start llama-server in background
    print(f"Starting llama-server with model: {gguf_model_path}")
    server_cmd = [llama_server_path, "-m", gguf_model_path, "-c", str(context_length)]
    server_process = subprocess.Popen(server_cmd)
    print(f"Started llama-server with PID: {server_process.pid}")

    try:
        # Give the server a moment to start up
        time.sleep(10)
        engine = EngineWrapper(
            api_key="Notused!We are local",
            base_url="http://127.0.0.1:8080/v1",
            mode="api",
            model="itmattersnot",
        )
        stop_token = get_stop_tokens(template)
        assistant_prefix = get_assistant_prefix(template)

        print("Your stop token is:")
        print(stop_token)
        print("Your assistant prefix is:")
        print(repr(assistant_prefix))

        app = FastAPI(title="Augmentoolkit Custom Model API Server", version="0.1.0")

        # Add CORS middleware to handle cross-origin requests from the frontend
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify exact origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "message": "RAG Chat Server is running"}

        @app.post("/generate")
        async def submit_chat_to_server(request: ChatRequest):
            messages = request.messages

            # first, if messages contains a system prompt, append the prompt
            # else, create it

            if len(messages) == 0:
                raise HTTPException(400, "Empty messages!")

            current_rag_chunks_for_prompt = (
                []
            )  # This list will be modified if shrinking occurs
            detailed_rag_objects_for_response = (
                []
            )  # For reporting original retrieved chunks
            rag_source_files_used = []  # Unique source file paths from RAG

            usr_msg = messages.pop()
            usr_string = usr_msg["content"]

            assert usr_msg["role"] == "user"

            if collection:
                retrieved_chunks = retrieve_relevant_chunks(
                    collection, usr_string, top_k
                )
                if retrieved_chunks:
                    current_rag_chunks_for_prompt = copy.deepcopy(
                        retrieved_chunks
                    )  # [chunk.copy() for chunk in retrieved_chunks] # Work with a copy for potential modification
                    for (
                        chunk
                    ) in (
                        current_rag_chunks_for_prompt
                    ):  # Populate detailed_rag_objects_for_response with original data
                        detailed_rag_objects_for_response.append(
                            {
                                "file_path": chunk["source_file_path"],
                                "char_offsets": chunk["char_offsets"],
                                "original_question_trigger": chunk[
                                    "original_question_trigger"
                                ],
                                # "text": chunk["text"] # Optionally include text if needed by frontend
                            }
                        )
                        if chunk["source_file_path"] not in rag_source_files_used:
                            rag_source_files_used.append(chunk["source_file_path"])
                    logger.info(
                        f"RAG: Retrieved {len(current_rag_chunks_for_prompt)} initial chunks for query: '{usr_string[:50]}...'"
                    )

            # System prompt has RAG added to it. So yes we do the systemprompt building first then we truncatem essagess after.

            if current_rag_chunks_for_prompt:
                rag_context_str = stringify_rag_chunks(current_rag_chunks_for_prompt)
                system_content_with_rag = f"{prompt}\n{rag_context_str}"

            current_token_count = count_tokens(system_content_with_rag)
            logger.info(
                f"Initial prompt token count (not counting past messages): {current_token_count}"
            )

            if current_token_count > context_length and current_rag_chunks_for_prompt:
                logger.info(
                    f"Token count {current_token_count} > {context_length}. Attempting to shrink RAG context."
                )
                for i in range(max_shrink_iterations):
                    if not current_rag_chunks_for_prompt:
                        logger.info("No more RAG chunks to shrink.")
                        break

                    # Find largest chunk by text length in the current list for the prompt
                    largest_chunk_idx, max_len = -1, -1
                    for idx, chunk_data in enumerate(current_rag_chunks_for_prompt):
                        if len(chunk_data["text"]) > max_len:
                            max_len = len(chunk_data["text"])
                            largest_chunk_idx = idx

                    if (
                        largest_chunk_idx == -1 or max_len <= 0
                    ):  # No suitable chunk to shrink
                        logger.info("Could not find a RAG chunk to shrink further.")
                        break

                    chunk_to_shrink = current_rag_chunks_for_prompt[largest_chunk_idx]
                    original_text_len = len(chunk_to_shrink["text"])
                    shrunk_text = chunk_to_shrink["text"][: original_text_len // 2]

                    logger.info(
                        f"Shrinking RAG chunk (idx {largest_chunk_idx}, original len {original_text_len}) from source '{chunk_to_shrink['source_file_path']}' to new len {len(shrunk_text)}."
                    )
                    chunk_to_shrink["text"] = shrunk_text

                    # Remove chunk if its text becomes empty after shrinking
                    if not chunk_to_shrink["text"].strip():
                        logger.info(
                            f"RAG chunk (idx {largest_chunk_idx}) became empty after shrinking, removing from prompt context."
                        )
                        current_rag_chunks_for_prompt.pop(largest_chunk_idx)

                    if current_rag_chunks_for_prompt:
                        rag_context_str = stringify_rag_chunks(
                            current_rag_chunks_for_prompt
                        )
                        system_content_with_rag = f"{prompt}\n{rag_context_str}"

                    current_token_count = count_tokens(system_content_with_rag)
                    logger.info(
                        f"Token count after RAG shrink iteration {i+1}: {current_token_count}"
                    )

                    if current_token_count <= context_length:
                        logger.info("Token count within limit after RAG shrinking.")
                        break
                if current_token_count > context_length:
                    logger.warning(
                        f"Token count still {current_token_count} after {max_shrink_iterations} RAG shrink attempts."
                    )
                    # possibly error here? Maybe? Hmm...

            current_messages = copy.deepcopy(messages)

            system_message_dict = {"role": "system", "content": system_content_with_rag}
            if current_messages and current_messages[0]["role"] == "system":
                current_messages[0] = system_message_dict
            else:
                current_messages.insert(0, system_message_dict)
                # WARNING your system messages get obliterated with this api

            ## NOTE Current up to here

            # Traditional token counting happens AFTER we do
            total_tokens = count_tokens(usr_string)
            for msg in current_messages:
                msg_tokens = count_tokens(msg["content"])
                total_tokens = total_tokens + msg_tokens

            shown_messages = current_messages.copy()

            while total_tokens >= context_length:
                if (
                    not len(shown_messages) == 1
                ):  # if we are down to the system prompt and the most recent user message and are over, then we have a dire problem
                    removed_message = shown_messages.pop(
                        1
                    )  # remove the message after the system prompt
                    removed_tokens = count_tokens(removed_message["content"])
                    total_tokens = total_tokens - removed_tokens
                else:
                    print(
                        f"\n\nNO MESSAGE SENT -- User message too long, user message + system message = {total_tokens} which is > {context_length}\n\n"
                    )
                    raise HTTPException(400, "Too long user message!")

            usr_message = {"role": "user", "content": usr_string}
            shown_messages.append(usr_message)
            message_string = format_messages_into_string(
                shown_messages, prompt_template=template
            )

            # Add assistant prefix for generation prefilling
            prefilled_prompt = message_string + assistant_prefix

            response, timeout = await engine.submit_completion(
                prompt=prefilled_prompt,
                sampling_params={
                    "temperature": 0.4,  # yes, it's not greedy. We want a good token probability distribtion for quality outputs.
                    "min_p": 0.3,
                    "top_p": 0.9,
                    "stop": stop_token,
                },  # sampling params should really be a list of dicts param_name: value. Because that way you could control the order in a nice way.
                return_completion_only=True,
            )

            return response  # the string

        set_progress(
            task_id=task_id,
            progress=1.0,
            message="Server is running! Navigate over to the chat window and you can interact with your model.",
        )

        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, use Server API
            config = uvicorn.Config(app=app, port=port, host="0.0.0.0")
            server = uvicorn.Server(config)
            await server.serve()
        except RuntimeError:
            # No event loop running, use the synchronous API
            uvicorn.run(app, port=port, host="0.0.0.0")
    finally:
        # Clean up the llama-server process
        if server_process:
            print("Terminating llama-server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
                print("llama-server terminated gracefully")
            except subprocess.TimeoutExpired:
                print("Force killing llama-server...")
                server_process.kill()
                server_process.wait()
                print("llama-server force killed")
