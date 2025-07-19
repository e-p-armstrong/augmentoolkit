import os
import nltk
import shutil
import logging
import hashlib

# Configure onnxruntime for SLURM environments before any other imports that might use it
if "SLURM_JOB_ID" in os.environ:    
    try:
        import onnxruntime as ort
        import multiprocessing

        _original_session = ort.InferenceSession

        def patched_session(*args, **kwargs):
            num_threads = min(multiprocessing.cpu_count(), 16)
            # Thread-Settings setzen, falls nicht vorhanden
            if 'sess_options' not in kwargs:
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = num_threads
                opts.intra_op_num_threads = num_threads
                kwargs['sess_options'] = opts
            else:
                opts = kwargs['sess_options']
                opts.inter_op_num_threads = num_threads
                opts.intra_op_num_threads = num_threads

            print(f"[Patch] ONNX session mit {num_threads} Threads")
            return _original_session(*args, **kwargs)

        class PatchedInferenceSession(_original_session):
            def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
                num_threads = min(multiprocessing.cpu_count(), 16)

                if sess_options is None:
                    sess_options = ort.SessionOptions()
                sess_options.inter_op_num_threads = num_threads
                sess_options.intra_op_num_threads = num_threads

                print(f"[Patch] Init ONNX InferenceSession with {num_threads} threads")

                super().__init__(
                    path_or_bytes,
                    sess_options=sess_options,
                    providers=providers,
                    provider_options=provider_options,
                    **kwargs
                )

        # Overwrite function which is used by chroma db
        ort.InferenceSession = PatchedInferenceSession

        
    except ImportError:
        # onnxruntime not available, which is fine
        pass

from tqdm import tqdm
import re
import sys

from augmentoolkit.generation_functions.cleanup import cleanup_dir
from augmentoolkit.generation_functions.hashing_and_ordering import hash_input_list
from augmentoolkit.generation_functions.majority_vote_step import MajorityVoteStep
from augmentoolkit.generation_functions.one_to_many_step import OneToManyStep
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
from augmentoolkit.utils.cost_estimation_logging import (
    calculate_pipeline_cost_efficiency,
)
from augmentoolkit.utils.observers import (
    create_input_token_counter,
    create_log_observer,
    create_output_token_counter,
)
from augmentoolkit.utils.work_with_output_files import recombine_many_to_one, save_data
from generation.core_components.chunking import (
    chunk_text_list,
    count_tokens,
    count_total_tokens,
    read_and_chunk_text,
    subset_text_list,
)
from generation.core_components.filter_chunks import (
    filter_out_failed_items,
    filter_out_failed_items_dict,
)
from generation.core_components.meta_datagen import create_meta_dataset
from generation.core_components.setup_components import (
    make_relative_to_self,
    setup_semaphore_and_engines,
)
from generation.core_pipelines.recall_multiple_sources.multi_source_helpers import (
    extract_questions_from_response,
    extract_reasoning_from_context_check,
    judge_paragraph_processor,
    parse_answer_accuracy_validation,
    parse_answer_relevancy_validation_step,
    parse_validation_step,
    save_conversations,
    save_plain_qatuples,
    scrape_text_read_files_manually,
    conversation_generator_step,
)
from redis_config import set_progress

nltk.download("punkt", quiet=True)
from augmentoolkit.generation_functions.process_multiturn_functions import (
    extract_conversation,
)
import augmentoolkit.utils.create_pretraining_set
import augmentoolkit.utils.sentence_chunking_algorithm
from augmentoolkit.utils.parse_bool import parse_bool
import asyncio
import traceback

import augmentoolkit.utils.group_by_text

# RAG-enabled variant


def filter_the_text(q_or_a):
    list_of_bad_strings = [
        # " the text",
        "according to the text",
        "as stated in",
        "explicitly stated",
        "as defined in",
        "given text",
        "provided information",
        "the text states",
    ]
    if any(bad_string in q_or_a for bad_string in list_of_bad_strings):
        return False
    return True


async def generate_multi_source_dataset(
    completion_mode,
    chunk_size,
    use_gutenberg,
    start_url,
    max_books,
    max_failures,
    hub_path,
    private,
    push_to_hub,
    input_dir,
    prompts,
    default_prompts,
    use_stop,
    do_not_use_system_prompts,
    final_assistant_prompts_no_rag,
    final_assistant_prompts_rag,
    rag_failure_percentage,
    items_per_conversation,
    concurrency_limit,
    small_model,
    small_api_key,
    small_base_url,
    small_mode,
    large_model,
    large_api_key,
    large_base_url,
    large_mode,
    use_subset,
    subset_size,
    output_dir,
    cost_per_million_small_input,
    cost_per_million_small_output,
    cost_per_million_large_input,
    cost_per_million_large_output,
    *args,
    cite_sources_at_end=True,
    read_files_manually=True,  # only set to false if it is being used as a node, we use kwargs to escape the default problem of clunkiness
    text_chunks_passed_in=[],  # also having node-based things means we need to return something doesn't it. May need to add additional args here. Do we assume that it's a proper dict? No?
    do_meta_datagen=False,
    meta_datagen_keys=[],
    meta_datagen_extras=[],
    chunking_output_dir=None,
    cleanup_embedding_dir=False,
    task_id=None,  # The good thing is that the task id requirement does not complicate the execution of pipelines inside other pipelines, and you don't need to think about THEIR progress as a part of THIS pipeline's progress. Thoug if we eid have away of peeking that progress it might be helpful since we coulddo th same with pipelinestep executions... no, because the policy is to favor API simplicity over fine control to make adoption/pipeline creation easier. Also the goal of the client is to make using this easier too. So gotchas like changing the node but not the config etc., editing things and mistakes -- these can be watched for and caught. Whether the path exists can be caught and shown as a specific warning. How will we record the rate oferrors? Simple, I guess we could how the % of things that made it to the end compared to how many we started with to get the percent.
    seed=1048596,
    uncap_retrieved_doc_length = False,
    **kwargs,
):

    prompts = make_relative_to_self(prompts)
    default_prompts = make_relative_to_self(default_prompts)

    print("UNUSED KWARGS!")
    print(kwargs)

    ### Definition of all pipeline steps must happen in here. Since use_filenames changes prompt paths significantly and it's an argument now, not a constant that can easily be shared between files. So we must define based on it.

    judgement_prompt_path = "judge_paragraph_filenames"

    judgement_regex = re.compile(
        r"Reasoning and thought process \(reason intelligently\):(.+)",
        re.DOTALL | re.IGNORECASE,
    )

    filter_all_questions_step = PipelineStep(
        prompt_path=judgement_prompt_path,
        regex=judgement_regex,
        sampling_params={
            "max_tokens": 1450,
            # "min_p": 0.4,
            "stop": [
                "### Response",
                "\n\n\n\n\n\n\n\n\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
            "temperature": 0.2,
        },
        output_processor=judge_paragraph_processor,
        result_key="judged_worthy_for_questions",
        output_file="judge_paragraph",
        details_key="judgement_details",
    )
    ### NOTE end definitions of pipeline steps

    ### NOTE might be a good idea for me to make a "delete key from datagen dict" function to help clean things up at certain points? Ah but that doesn't account for saved values, modifying it in memory does nothing if it's already on disk. hmm tough question.

    small_token_counter = {
        "input_tokens": 0,
        "input_cost": 0.0,
        "output_tokens": 0,
        "output_cost": 0.0,
        "name": "Small model",
    }
    large_token_counter = {
        "input_tokens": 0,
        "input_cost": 0.0,
        "output_tokens": 0,
        "output_cost": 0.0,
        "name": "Large model",
    }

    run_task_with_limit, engine_wrapper, engine_wrapper_large, _ = (
        setup_semaphore_and_engines(
            concurrency_limit,
            small_model,
            small_api_key,
            small_base_url,
            small_mode,
            large_model,
            large_api_key,
            large_base_url,
            large_mode,
            engine_input_observers=[
                create_input_token_counter(
                    counter=small_token_counter,
                    cost_per_million=cost_per_million_small_input,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "small_model_tokens.json"
                    ),
                )
            ],
            engine_output_observers=[
                create_log_observer(output_dir, do_meta_datagen),
                create_output_token_counter(
                    counter=small_token_counter,
                    cost_per_million=cost_per_million_small_output,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "small_model_tokens.json"
                    ),
                ),
            ],
            large_engine_input_observers=[
                create_input_token_counter(
                    counter=large_token_counter,
                    cost_per_million=cost_per_million_large_input,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "large_model_tokens.json"
                    ),
                )
            ],
            large_engine_output_observers=[
                create_log_observer(output_dir, do_meta_datagen),
                create_output_token_counter(
                    counter=large_token_counter,
                    cost_per_million=cost_per_million_large_output,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "large_model_tokens.json"
                    ),
                ),
            ],
        )
    )

    semaphore = asyncio.Semaphore(concurrency_limit)

    async def run_task_with_limit(task, timeout=60):
        async with semaphore:
            try:
                return await asyncio.wait_for(task, timeout=timeout)
            except asyncio.TimeoutError:
                print("[Timeout] A task exceeded the timeout limit.")
                return None
            except Exception as e:
                print(f"[Error] Exception in task: {e}")
                return None

    # notably, to be used as a node these things need to take their sentence chunks as input, not an input dir path. Which means that this step wouldn't actually fire. We won't be making a pretraining dataset in the original pipeline anymore since that's handled by repvar when run as part of a larger system.
    # We need to engineer better how to make pipelines vs nodes work. When in node mode, these things will behave different. When in pipeline mode they'll take a certain set of args vs when in node mode they'll take a certain other set. How do we represent this best? Perhaps ask AI. What is the cleanest and best way to engineer this such that pipelines can fulfill both the role of pipeline and node 1) without cluttering the code too much, 2) while maintaining the nice modularity of everything so far?
    # Basically the key annoying thing is that when it's a pipeline we use paths to things like inputs, etc. But as nodes we want to pass arguments in, like a list of sentence chunks. We simply need a flag and then alternate kwargs I suppose.
    # Even prompts might not be relative to self, but instead passed in, an absolute path when this is used as a node
    # in fact I think it will be that way, at least by default for me. Because that way you have more control and it is more explicit and it's all centralized in hte single PIPELINE that you run you don't need to manage multiple configs and folders that way.
    # well but at the same time that means that things can't be seamlessly used across things, like filter is.
    # ARGH tough. Well we should support both and just clearly indicate which is which.
    # The options being, either have your nodes use the prompts in their folder, or pass in the prompts folder path for your nodes to use.

    set_progress(
        task_id=task_id,
        progress=0.0,
        message="Pipeline starting; reading and chunking files",
    )

    if read_files_manually:
        print("Using config...")
        if use_gutenberg:
            scrape_text_read_files_manually(
                start_url=start_url,
                max_books=max_books,
                max_failures=max_failures,
                input_dir=input_dir,
            )

        sentence_chunks = read_and_chunk_text(
            input_dir=input_dir,
            chunk_size=chunk_size,
            use_subset=False,
            subset_size=subset_size,
            keep_folder_structure=True,
            output_dir=chunking_output_dir if chunking_output_dir else output_dir,
            seed=seed,
        )

    else:
        print("Using text chunks passed in...")
        print("LEN OF CHUNKS BEFORE")
        print(len(text_chunks_passed_in))
        sentence_chunks = chunk_text_list(
            text_chunks_passed_in,
            chunk_size,
            keep_folder_structure=True,
            input_dir=input_dir,
            output_dir=chunking_output_dir if chunking_output_dir else output_dir,
        )
        print("!!LEN OF SENTENCE CHNKS")
        print(len(sentence_chunks))

    set_progress(
        task_id,
        progress=0.1,
        message="Files read and chunked! Proceeding with RAG indexing",
    )

    ### RAG SETUP

    # Set up RAG collection from sentence chunks
    import chromadb
    from chromadb.utils import embedding_functions

    # Determine base output directory
    base_output_dir = chunking_output_dir if chunking_output_dir else output_dir

    # Use a stable hash algorithm instead of the built-in hash()
    hash_object = hashlib.sha256(input_dir.encode("utf-8"))
    hash_value = hash_object.hexdigest()[:16]  # Use a portion of the hex digest
    chunk_hash_suffix = hash_value  # Store the hash suffix

    # Construct the dynamic ChromaDB directory path
    chroma_db_dir_name = f"chroma_db_{chunk_hash_suffix}"
    chroma_db_path = os.path.join(
        base_output_dir, chroma_db_dir_name
    )  # Store the full path
    print(f"Using ChromaDB persistent path: {chroma_db_path}")

    print("Initializing RAG system...")
    # Initialize ChromaDB client with the dynamic path
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)

    # Use default embedding function
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    # Create collection name based on hash of input chunks (can keep this, doesn't hurt)
    # Note: Hashing the full string representation can be slow for large datasets.
    # try:
    #     # Using the original hash method for now
    #     hash_input_str = str(sentence_chunks)
    # except Exception as e:
    #     logging.warning(f"Could not stringify sentence_chunks for hashing, falling back to simple repr: {e}")
    #     hash_input_str = repr(sentence_chunks) # Fallback if stringification fails

    # hash_value = str(hash(hash_input_str))
    collection_name = f"sentence_chunks_{chunk_hash_suffix}"
    # print(f"Target RAG collection name: {collection_name}")
    print(f"Target RAG collection name inside DB: {collection_name}")

    collection_exists = False
    collection_populated = False
    collection = None
    # Calculate expected count *before* trying to get/create
    expected_doc_count = sum(
        1 for chunk in sentence_chunks if chunk.get("text", "").strip()
    )

    try:
        print(f"Checking for existing collection '{collection_name}'...")
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,  # Ensure embedding function matches
        )
        collection_exists = True
        print(f"Found existing collection '{collection_name}'.")

        current_count = collection.count()
        if current_count > 0 and current_count == expected_doc_count:
            print(
                f"Collection '{collection_name}' is already populated with {current_count} documents (matching expected {expected_doc_count}). Skipping addition."
            )
            collection_populated = True
        elif current_count > 0:
            logging.warning(
                f"Collection '{collection_name}' exists but has {current_count} documents, expected {expected_doc_count}. Will re-populate."
            )
            # Delete the mismatched collection to ensure clean state
            chroma_client.delete_collection(name=collection_name)
            print(f"Deleted existing mismatched collection '{collection_name}'.")
            collection_exists = False  # Reset flag so it gets created below
        else:
            print(
                f"Collection '{collection_name}' exists but is empty (count: {current_count}). Proceeding with population."
            )

    except Exception as e:
        # Handle case where collection doesn't exist or other error during get
        print(
            f"Collection '{collection_name}' not found or error getting it: {e}. Will create and populate."
        )
        # collection_exists remains False

    if not collection_populated:
        if not collection_exists:  # Only create if get_collection failed
            print(f"Creating collection '{collection_name}'...")
            collection = chroma_client.get_or_create_collection(
                name=collection_name, embedding_function=embedding_function
            )

        # Add sentence chunks to collection (existing logic, with minor safety improvements)
        documents = []
        metadatas = []
        ids = []
        doc_index = 0  # Use a separate index for IDs

        print(
            f"Adding {expected_doc_count} chunks to the RAG collection '{collection_name}'..."
        )

        for chunk in tqdm(
            sentence_chunks,
            desc=f"Adding chunks to RAG '{collection_name}'",
            unit="chunk",
        ):
            chunk_text = chunk.get("text", "")  # Use .get for safety
            if chunk_text.strip():  # Ensure we only add non-empty chunks
                documents.append(chunk_text)
                # Use .get for metadata safety
                metadatas.append({"source": chunk.get("metadata", "unknown_source")})
                ids.append(f"doc_{doc_index}")
                doc_index += 1

                # Add in batches
                if len(documents) >= 100:
                    # print(f"Vectorizing and adding batch of {len(documents)} documents...") # Reduced verbosity
                    try:
                        collection.add(
                            documents=documents, metadatas=metadatas, ids=ids
                        )
                    except Exception as add_error:
                        logging.error(
                            f"Error adding batch to ChromaDB collection '{collection_name}': {add_error}"
                        )
                        # Consider whether to break or continue here
                    finally:  # Ensure lists are cleared even if add fails
                        documents = []
                        metadatas = []
                        ids = []

        # Add any remaining documents
        if documents:
            # print(f"Vectorizing and adding final batch of {len(documents)} documents...") # Reduced verbosity
            try:
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
            except Exception as add_error:
                logging.error(
                    f"Error adding final batch to ChromaDB collection '{collection_name}': {add_error}"
                )

    try:
        final_count = collection.count()
        print(f"RAG collection '{collection_name}' ready with {final_count} chunks.")
        if (
            final_count != expected_doc_count and not collection_populated
        ):  # Check if population was attempted but failed
            logging.warning(
                f"Final document count ({final_count}) in collection '{collection_name}' does not match expected count ({expected_doc_count}) after population attempt."
            )
        elif collection_populated and final_count != expected_doc_count:
            logging.error(
                f"Loaded existing collection '{collection_name}' but count ({final_count}) mismatch expected ({expected_doc_count}). This shouldn't happen."
            )
    except Exception as e:
        logging.error(
            f"Could not verify final count for collection '{collection_name}': {e}"
        )

    set_progress(
        task_id,
        progress=0.5,
        message="Chunks indexed for RAG retrieval! Proceeding with Chunk Filtering!",
    )

    # Function to retrieve relevant chunks based on a query
    async def retrieve_rag_chunks(query, top_k=3, timeout_seconds=30):
        """
        Retrieve chunks from the RAG collection that are relevant to the query

        Args:
            query: The text to find relevant chunks for
            top_k: Number of chunks to retrieve (default 3)
            timeout_seconds: Timeout for the query operation (default 30)

        Returns:
            A list of dictionaries with text and metadata
        """
        try:
            # Wrap the synchronous collection.query call with a timeout
            results = await asyncio.wait_for(
                asyncio.to_thread(collection.query, query_texts=[query], n_results=top_k),
                timeout=timeout_seconds
            )

            chunks = []
            if results and results["documents"] and len(results["documents"]) > 0:
                logging.info(
                    f"RAG Query for '{query[:50]}...': Retrieved {len(results['documents'][0])} documents."
                )  # Log retrieval success and count
                for i, doc in enumerate(results["documents"][0]):

                    chunks.append(
                        {
                            "text": doc if uncap_retrieved_doc_length else doc[:6000],
                            "metadata": results["metadatas"][0][i]["source"],
                        }
                    )
            else:
                logging.warning(
                    f"RAG Query for '{query[:50]}...': Retrieved 0 documents."
                )  # Log retrieval failure

            return chunks
        except asyncio.TimeoutError:
            logging.error(f"RAG Query for '{query[:50]}...' timed out after {timeout_seconds} seconds.")
            print(f"RAG query timed out after {timeout_seconds} seconds for query: {query[:50]}...")
            return []
        except Exception as e:
            print(f"Error retrieving RAG chunks: {str(e)}")
            import traceback

            traceback.print_exc()
            return []

    # subset the sentence chunks
    if use_subset:
        print("SUBSETTING TEXT CHNKS")
        sentence_chunks = subset_text_list(sentence_chunks, subset_size, seed=seed)

    total_tokens = count_total_tokens(sentence_chunks)

    sentence_dict = hash_input_list(input_list=sentence_chunks, key_to_hash_with="text")

    print(
        "\n\n\n=================== FILTERING QUESTIONS ============================ \n\n\n"
    )
    await filter_all_questions_step.execute_pipeline(
        input_dict=sentence_dict,
        engine_wrapper=engine_wrapper,
        rtwl=run_task_with_limit,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
        include_details=do_meta_datagen,
    )

    filter_out_failed_items_dict(
        sentence_dict, key_to_check="judged_worthy_for_questions"
    )

    set_progress(
        task_id, progress=0.6, message="Chunks filtered! Proceeding with QA Generation!"
    )

    # for key, value in sentence_dict.items():
    #     # Use RAG to extract chunks related to the text
    #     related_chunks = retrieve_rag_chunks(value["text"], top_k=3)

    #     # Skip self-referential chunks (same chunk retrieved for itself)
    #     related_chunks = [chunk for chunk in related_chunks
    #                       if chunk["text"] != value["text"]]

    #     # Store the RAG context with the item
    #     if related_chunks:
    #         # Format the RAG context
    #         rag_context = ""
    #         for i, chunk in enumerate(related_chunks):
    #             rag_context += f"\nRelated Text {i+1}, from source {chunk['metadata']}:\n\"\"\"\n{chunk['text']}\n\"\"\"\n-----------\n"

    #         # Add the RAG context to the item
    #         value["rag_context"] = rag_context
    #         value["related_chunks"] = related_chunks
    #     else:
    #         value["rag_context"] = ""
    #         value["related_chunks"] = []

    prompt_path_qatuples_gen = "qatuples_gen_filenames"

    class QGenStep(OneToManyStep):
        async def process_input_data(self, input_data):
            # print(f"QGenStep: Processing item with text starting: {input_data['text'][:50]}...") # Log item start
            # print("QGenStep: Retrieving RAG chunks...")
            related_chunks = await retrieve_rag_chunks(input_data["text"], top_k=3)
            # print(f"QGenStep: RAG retrieval complete. Found {len(related_chunks)} related chunks (excluding self).")

            # Skip self-referential chunks (same chunk retrieved for itself)
            related_chunks = [
                chunk for chunk in related_chunks if chunk["text"] != input_data["text"]
            ]  # Corrected: value -> input_data

            # Store the RAG context with the item
            if related_chunks:
                # Format the RAG context
                rag_context = ""
                for i, chunk in enumerate(related_chunks):
                    rag_context += f"\nRelated Text {i+1}, from source {chunk['metadata']}:\n\"\"\"\n{chunk['text']}\n\"\"\"\n-----------\n"

                # Add the RAG context to the item
                input_data["rag_context"] = rag_context
                input_data["related_chunks"] = related_chunks
            else:
                input_data["rag_context"] = ""
                input_data["related_chunks"] = []

            print("QGenStep: Finished processing item.")  # Log item end
            return input_data, {}

    question_generation_step = QGenStep(
        prompt_path=prompt_path_qatuples_gen,
        sampling_params={
            "max_tokens": 2000,
            "stop": [
                "### Response",
                "\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
            "temperature": 0.8,
            "top_p": 1,
        },
        output_processor=extract_questions_from_response,
        result_key="qa_tuples",
        max_retries=3,
        log_full_outputs=False,
        output_file="factual_questions",
        details_key="factual_questions_details",
        input_file="judge_paragraph",
    )

    # Generate questions
    print(
        "generate_multi_source_dataset: Starting question_generation_step.execute_pipeline"
    )  # Log before execution
    questions_dict = await question_generation_step.execute_pipeline(
        input_dict=sentence_dict,
        engine_wrapper=engine_wrapper_large,
        rtwl=run_task_with_limit,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
        include_details=do_meta_datagen,
    )
    print(
        f"generate_multi_source_dataset: Finished question_generation_step.execute_pipeline. Result count: {len(questions_dict)}"
    )  # Log after execution

    print(
        "generate_multi_source_dataset: Starting processing of questions_dict"
    )  # Log loop start
    for key, value in questions_dict.items():
        # print("Iterating over items")
        questions_dict[key]["question"] = value["qa_tuples"]["question"]
        questions_dict[key]["answer"] = value["qa_tuples"]["answer"]
        # print(value)

    save_data(
        questions_dict, output_dir, "factual_questions"
    )  # could this be the problem? No. Because th4e operation should save the new key to the judged worthy for quewstions dict list thing. Not this one.

    print(
        "generate_multi_source_dataset: Finished processing questions_dict"
    )  # Log loop end

    set_progress(
        task_id,
        progress=0.95,
        message="Questions and answers generated! Proceeding with dataset save.",
    )

    for key in questions_dict:
        # Clean up additional keys we no longer need
        keys_to_delete = [
            "judged_worthy_for_questions",
            "qa_tuples",
            "question_validation_votes",
            "answer_rel_votes",
            "answer_acc_votes",
        ]
        for k in keys_to_delete:
            if k in questions_dict[key]:
                del questions_dict[key][k]

    qa_dicts_by_text = []
    text_hash_groups = {}

    # First pass - group by text hash
    text_hash_groups = recombine_many_to_one(questions_dict)

    plain_qa_list, simplified_rag_list = save_plain_qatuples(
        qa_dicts_by_text=text_hash_groups,
        final_assistant_prompts_rag=final_assistant_prompts_rag,
        final_assistant_prompts_no_rag=final_assistant_prompts_no_rag,
        rag_failure_percentage=rag_failure_percentage,
        do_not_use_system_prompts=do_not_use_system_prompts,
        items_per_conversation=items_per_conversation,
        output_dir=output_dir,
        hub_path=hub_path,
        push_to_hub=push_to_hub,
        cite_sources_at_end=cite_sources_at_end,
    )

    cleanup_dir(output_dir=output_dir)

    calculate_pipeline_cost_efficiency(
        total_input_tokens=total_tokens,
        token_counters=[small_token_counter, large_token_counter],
    )

    # remove the judgement_details key from each dict in questions_dict
    for key, value in questions_dict.items():
        if "judgement_details" in value:
            del value["judgement_details"]

    if cleanup_embedding_dir:  # THIS is why multi turn was slow to resume.
        # Remove the specific chroma_db directory for this run, using the variable
        # chroma_db_path = os.path.join(output_dir, "chroma_db") # OLD HARDCODED PATH
        if os.path.exists(chroma_db_path) and os.path.isdir(chroma_db_path):
            try:
                print(f"Removing ChromaDB directory: {chroma_db_path}")
                shutil.rmtree(chroma_db_path)
                print("ChromaDB directory removed successfully.")
            except OSError as e:
                print(f"Error removing ChromaDB directory {chroma_db_path}: {e}")
                traceback.print_exc()

    if do_meta_datagen:
        create_meta_dataset(
            data_dicts=[questions_dict, sentence_dict],
            meta_datagen_keys=meta_datagen_keys,
            meta_datagen_extras=meta_datagen_extras,
            input_processors=[],
            output_dir=os.path.join(output_dir, "meta_datagen"),
        )

    set_progress(task_id, progress=1.0, message="Pipeline Complete!")

    return plain_qa_list, simplified_rag_list

    # and now we go and save with the save plain qalist function

    # there we go

    # and conversation generation, I suppose is fine, even if I never use it anymore
    # we want decent backwards compatibility with the prompts after all
    # plus it's a great step to put things like translation into
    # Evan include your translation pipeline with this newest release
    # it'll be great

    # context repair
    # basically we'll have a "repaired context" key
    # and the output processor decides what goes in there based on the judgement:
    # if reword, it puts the reworded thing
    # if pass, it puts the original thing
    # if fail, it puts false, and we pass this through a filter out failed items dict thing. That thing only checks truthy or falsey so it will probably work out, I think.
