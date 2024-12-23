import sys
import os
import nltk
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import yaml
import glob
import asyncio
import traceback
import logging

# Ensure NLTK's punkt tokenizer is available
nltk.download('punkt', quiet=True)

# Import necessary functions and classes from augmentoolkit
from augmentoolkit.generation_functions.process_multiturn_functions import extract_conversation
import augmentoolkit.utils.create_pretraining_set
import augmentoolkit.utils.sentence_chunking_algorithm
from augmentoolkit.utils.parse_bool import parse_bool
import augmentoolkit.utils.group_by_text

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)
# Add the script directory and its parent to the Python path
sys.path.append(script_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define a text filtering function
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

# Define a function to process a single file
def process_file(source_text, chunk_size):
    """
    Process a single file: perform sentence chunking and return the chunks and content.

    Args:
        source_text (str): Path to the source text file.
        chunk_size (int): The size of each chunk as defined in CHUNK_SIZE.

    Returns:
        tuple: (chunks, content)
    """
    try:
        chunks, content = augmentoolkit.utils.sentence_chunking_algorithm.sentence_chunking_algorithm(
            source_text, chunk_size
        )
        return chunks, content
    except Exception as e:
        print(f"Error processing {source_text}: {e}")
        return [], ""

# Define a function to write batches to JSONL files
def write_batches(pretraining_dir, batch, file_index, prefix="pretraining"):
    """
    Write a batch of data to a JSONL file.

    Args:
        pretraining_dir (str): Directory where pretraining files are stored.
        batch (list): List of JSON-serializable objects to write.
        file_index (int): Index of the current pretraining file.
        prefix (str): Prefix for the pretraining files.

    Returns:
        int: Number of lines written.
    """
    filename = f"{prefix}.jsonl" if file_index == 1 else f"{prefix}_{file_index}.jsonl"
    filepath = os.path.join(pretraining_dir, filename)
    with open(filepath, 'a', encoding='utf-8') as f:
        for item in batch:
            json_line = json.dumps(item)
            f.write(json_line + '\n')
    return len(batch)

# Main asynchronous function
async def main():
    # Load configuration
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")  # Default to 'config.yaml' if not set
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set up logging
    LOG_LEVEL = logging.INFO
    logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create output directory if it doesn't exist
    output_path = config["PATH"]["OUTPUT"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Configuration Parameters
    DOUBLE_CHECK_COUNTER = int(config["SYSTEM"]["DOUBLE_CHECK_COUNTER"])
    USE_SUBSET = parse_bool(config["SYSTEM"]["USE_SUBSET"])
    SUBSET_SIZE = int(config["SYSTEM"]["SUBSET_SIZE"])
    USE_FILENAMES = parse_bool(config["SYSTEM"]["USE_FILENAMES"])
    CONCURRENCY_LIMIT = int(config["SYSTEM"]["CONCURRENCY_LIMIT"])
    SMALL_BASE_URL = config["API"]["SMALL_BASE_URL"]
    SMALL_MODEL = config["API"]["SMALL_MODEL"]
    SMALL_API_KEY = config["API"]["SMALL_API_KEY"]
    SMALL_MODE = config["API"]["SMALL_MODE"]
    LARGE_BASE_URL = config["API"]["LARGE_BASE_URL"]
    LARGE_MODEL = config["API"]["LARGE_MODEL"]
    LARGE_API_KEY = config["API"]["LARGE_API_KEY"]
    LARGE_MODE = config["API"]["LARGE_MODE"]
    COMPLETION_MODE = parse_bool(config["SYSTEM"]["COMPLETION_MODE"])
    INPUT_FOLDER = config["PATH"]["INPUT"]
    PHASE_INDEX = int(config["PHASE"]["PHASE_INDEX"])
    WORK_IN_PHASES = parse_bool(config["PHASE"]["WORK_IN_PHASES"])
    SKIP_FILTER_CHUNKS = parse_bool(config["SKIP"]["FILTER_CHUNKS"])
    SKIP_REPAIR_QA_TUPLES = parse_bool(config["SKIP"]["REPAIR_QA_TUPLES"])
    CHUNK_SIZE = config["SYSTEM"]["CHUNK_SIZE"]
    USE_GUTENBERG = config["SCRAPING"]["USE_GUTENBERG"]
    START_URL = config["SCRAPING"]["START_URL"]
    MAX_BOOKS = config["SCRAPING"]["MAX_BOOKS"]
    MAX_FAILURES = config["SCRAPING"]["MAX_FAILURES"]
    SKIP_CONVERSATION_GENERATION = parse_bool(config["SKIP"]["CONVERSATION_GENERATION"])

    # Initialize Engine Wrappers
    import augmentoolkit.generation_functions as generation_functions  # This is the package directory
    from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper

    engine_wrapper = EngineWrapper(
        model=SMALL_MODEL,
        api_key=SMALL_API_KEY,
        base_url=SMALL_BASE_URL,
        mode=SMALL_MODE,
        # quantization="gptq" # modify if you want to do stuff with the aphrodite branch
    )
    
    engine_wrapper_large = EngineWrapper(
        model=LARGE_MODEL,
        api_key=LARGE_API_KEY,
        base_url=LARGE_BASE_URL,
        mode=LARGE_MODE,
        # quantization="gptq" # modify if you want to do stuff with the aphrodite branch
    )

    # Optional Gutenberg Scraping
    if USE_GUTENBERG:
        print("SCRAPING IS ON. BEGINNING GUTENBERG SCRAPE! This will modify your input folder.")
        from original import steps
        steps.scrape_text_using_config(start_url=START_URL, max_books=MAX_BOOKS, max_failures=MAX_FAILURES)

    # Collect Source Texts
    extensions = [".txt", ".md", ".pdf", ".docx", ".epub", ".html"]
    print(f"\n\n\nUSE FILENAMES: {USE_FILENAMES}")

    source_texts = []
    for extension in extensions:
        path = os.path.join(INPUT_FOLDER, "**", "*" + extension)
        source_texts += glob.glob(path, recursive=True)

    if source_texts:
        print(f"Found {len(source_texts)} source texts.")
    else:
        print(f"No source texts found in: {INPUT_FOLDER}")

    # Informational Message
    print(
        "\n\n\nIMPORTANT NOTE! Augmentoolkit prints a lot of stuff when it runs. Including tracebacks caused by model errors. Most errors are the result of the models, not the code, and any tracebacks you see were almost certainly handled. So: don't panic! You're gonna make it! Alright that's the end of this PSA. Happy dataset generation!\n\n\n"
    )

    # Set up rate-limit-conscious functions
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def run_task_with_limit(task):
        async with semaphore:
            return await task

    # Define multi_turn_convs_info_dir
    multi_turn_convs_info_dir = (
        os.path.join(config["PATH"]["OUTPUT"], "multi_turn_convs_info")
    )

    sys.path.append("./generation_functions")
    sys.path.append("./control_flow_functions")

    # Section g: Parallel Processing and Incremental Batch Writing
    print("Starting parallel processing of input files and writing pretraining.jsonl in batches...")

    # Define constants for batch processing
    BATCH_SIZE = 1000
    MAX_LINES_PER_FILE = 10000000

    pretraining_output_dir = os.path.join(config["PATH"]["OUTPUT"])
    pretraining_file_prefix = "pretraining"
    pretraining_file_extension = ".jsonl"

    # Ensure the output directory exists
    if not os.path.exists(pretraining_output_dir):
        os.makedirs(pretraining_output_dir)

    # Initialize variables for batch writing
    current_batch = []
    current_line_count = 0
    file_index = 1  # Start with pretraining.jsonl

    # Define the path for the pretraining files
    pretraining_file_path = os.path.join(pretraining_output_dir, f"{pretraining_file_prefix}{pretraining_file_extension}")

    # Remove existing pretraining files if they exist to prevent appending to old data
    for existing_file in glob.glob(os.path.join(pretraining_output_dir, f"{pretraining_file_prefix}*.jsonl")):
        os.remove(existing_file)

    # Use ProcessPoolExecutor to process files in parallel
    num_workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Prepare partial function with fixed chunk_size
        process_func = partial(process_file, chunk_size=CHUNK_SIZE)
        
        # Submit all tasks to the executor
        future_to_file = {executor.submit(process_func, source_text): source_text for source_text in source_texts}
        
        # Use tqdm to display progress
        for future in tqdm(as_completed(future_to_file), total=len(source_texts), desc="Processing and Chunking Files"):
            chunks, content = future.result()

            # Append content to batch
            if content:
                batch_item = {"content": content}
                current_batch.append(batch_item)
                current_line_count += 1

                # Check if batch size reached
                if len(current_batch) >= BATCH_SIZE:
                    # Write the current batch to the appropriate pretraining file
                    lines_written = write_batches(pretraining_output_dir, current_batch, file_index, prefix=pretraining_file_prefix)
                    current_batch = []  # Reset batch
                    current_line_count += lines_written

                    # Check if we need to switch to a new file
                    if current_line_count >= MAX_LINES_PER_FILE:
                        file_index += 1
                        current_line_count = 0  # Reset line count for new file
            
            # Handle chunks if needed (based on your original logic)
            for chunk in chunks:
                batch_item = {"chunk": chunk, "metadata": {"source": future_to_file[future]}}
                current_batch.append(batch_item)
                current_line_count += 1

                # Write batch if size reached
                if len(current_batch) >= BATCH_SIZE:
                    lines_written = write_batches(pretraining_output_dir, current_batch, file_index, prefix=pretraining_file_prefix)
                    current_batch = []  # Reset batch
                    current_line_count += lines_written

                    if current_line_count >= MAX_LINES_PER_FILE:
                        file_index += 1
                        current_line_count = 0  # Reset line count for new file

    # Write any remaining data in the last batch
    if current_batch:
        write_batches(pretraining_output_dir, current_batch, file_index, prefix=pretraining_file_prefix)

    print("Pretraining set created and written in batches successfully.")

    # PHASE 0 END
    print("\n\nCOMPLETED PHASE 0")
    if WORK_IN_PHASES and PHASE_INDEX == 0:
        sys.exit(0)

    #####

    # Control Flow
    import json

    generated_qa_dicts = []  # list of QA tuples that have been judged good

    # PHASE 1: Generate QA Dictionaries
    print("Starting Phase 1: Generating QA Dictionaries")
    tasks = [
        steps.generate_qadicts_from_para(
            idx,
            para,
            engine_wrapper_large=engine_wrapper_large,
            generated_qa_dicts=generated_qa_dicts,
        )
        for idx, para in enumerate(filtered_worthy_for_questions)
    ]
    limited_tasks_qgen = [run_task_with_limit(task) for task in tasks]
    for future in tqdm(asyncio.as_completed(limited_tasks_qgen), total=len(limited_tasks_qgen), desc="Generating QA Dicts"):
        await future

    # PHASE 1 END
    print("COMPLETED PHASE 1")
    if WORK_IN_PHASES and PHASE_INDEX == 1:
        print("EXITING DUE TO config.yaml SETTINGS AROUND PHASES; SET TO ONLY EXECUTE PHASE 1 RIGHT NOW")
        sys.exit(0)
    ####

    # PHASE 2: Vet QA Dictionaries
    print("Starting Phase 2: Vetting QA Dictionaries")
    vetted_qa_dicts = []
    qa_dicts_dir_checked = os.path.join(config["PATH"]["OUTPUT"], "qatuples_filtered")
    if not os.path.exists(qa_dicts_dir_checked):
        os.makedirs(qa_dicts_dir_checked)

    if generated_qa_dicts:
        print(f"First generated QA dict: {generated_qa_dicts[0]}")
    else:
        print("No generated QA dictionaries found.")
        sys.exit(1)

    tasks = [
        steps.vet_question_loop(
            question_answer_dict,
            question_group_id=question_answer_dict['question_group_id'],
            engine_wrapper=engine_wrapper,
            qa_dicts_dir=qa_dicts_dir_checked,
            vetted_qa_dicts=vetted_qa_dicts,
            double_check_counter=DOUBLE_CHECK_COUNTER,
            completion_mode=COMPLETION_MODE,
            logging_level=LOG_LEVEL,
        ) for question_answer_dict in generated_qa_dicts
    ]
    limited_tasks_q_validation = [run_task_with_limit(task) for task in tasks]
    for future in tqdm(asyncio.as_completed(limited_tasks_q_validation), total=len(limited_tasks_q_validation), desc="Vetting QA Dicts"):
        await future

    if WORK_IN_PHASES and PHASE_INDEX == 2:
        print("EXITING DUE TO config.yaml SETTINGS AROUND PHASES; SET TO ONLY EXECUTE PHASE 2 RIGHT NOW")
        sys.exit(0)

    print(
        "-------------- QUESTIONS CREATED ------------- STATS SO FAR (may be wrong if run was continued from interruption):"
    )
    nones = list(filter(lambda x: x is None, vetted_qa_dicts))
    print(f"Nones: {len(nones)}")
    print(f"Non-nones: {len(vetted_qa_dicts) - len(nones)}")
    print(f"Total: {len(vetted_qa_dicts)}")
    # filter out all None values
    vetted_qa_dicts = [qa for qa in vetted_qa_dicts if qa is not None]
    print("---------------- ONTO REVISION ------------------")

    # PHASE 3: Repair QA Tuples (Optional)
    if not SKIP_REPAIR_QA_TUPLES:
        print("Starting Phase 3: Repairing QA Tuples")
        tasks = [
            steps.repair_qatuple_context(
                idx,
                tup,
                engine_wrapper_large,
                vetted_qa_dicts,
            )
            for idx, tup in enumerate(vetted_qa_dicts)
        ]
        limited_tasks_qcorrection = [run_task_with_limit(task) for task in tasks]
        for future in tqdm(asyncio.as_completed(limited_tasks_qcorrection), total=len(limited_tasks_qcorrection), desc="Repairing QA Tuples"):
            await future
        print("-------------- QUESTIONS REVISED ------------- STATS SO FAR:")
        nones = list(filter(lambda x: x is None, vetted_qa_dicts))
        print(f"Nones: {len(nones)}")
        print(f"Non-nones: {len(vetted_qa_dicts) - len(nones)}")
        print(f"Total: {len(vetted_qa_dicts)}")
        # filter out all None values
        vetted_qa_dicts = [qa for qa in vetted_qa_dicts if qa is not None]
        print("---------------- ONTO EXAMPLES GENERATION-------------------")
    else:
        print("Skipping question repair")

    # Final Filtering
    vetted_qa_dicts = [qadict for qadict in vetted_qa_dicts if filter_the_text(qadict["question"]) and filter_the_text(qadict["answer"])]

    # Group QA dicts by text
    qa_dicts_by_text = augmentoolkit.utils.group_by_text.group_by_text(vetted_qa_dicts)
    
    print("Creating question generation training data...")
    steps.convert_revised_questions_to_question_generation_training(qa_dicts_by_text=qa_dicts_by_text, use_filenames=USE_FILENAMES)
    
    # Conversation Generation (Optional)
    if SKIP_CONVERSATION_GENERATION:
        print("Skipping conversation generation")
        steps.save_plain_qatuples(qa_dicts_by_text=qa_dicts_by_text)
    else:
        print("Starting conversation generation")
        multi_turn_convs = []

        tasks = [
            steps.create_conversation(
                idx,
                info,
                engine_wrapper_large,
                multi_turn_convs,
            )
            for idx, info in enumerate(qa_dicts_by_text)
        ]
        limited_tasks_convwriting = [run_task_with_limit(task) for task in tasks]
        for future in tqdm(asyncio.as_completed(limited_tasks_convwriting), total=len(limited_tasks_convwriting), desc="Creating Conversations"):
            await future

        print("Converting conversational data generations to training data")
        steps.convert_logging_to_dataset(input_pth=os.path.join("multi_turn_convs", "intermediate_generations"), output_pth="multi_turn_convs")
        
        # Make ShareGPT dataset
        steps.convert_directory_to_list(
            os.path.join(config["PATH"]["OUTPUT"], "multi_turn_convs", "saved_readable_generations")
        )
        
    # Yay! Now you have a dataset!
    
    # Load master list
    master_list_path = os.path.join(config["PATH"]["OUTPUT"], "master_list.jsonl")
    if os.path.exists(master_list_path):
        with open(master_list_path, "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        print(f"Master list not found at {master_list_path}.")
        data = []

    # Count GPT turns
    gpt_turns = 0        
    for entry in data:
        if not SKIP_CONVERSATION_GENERATION:
            conv = entry.get('conversation', '')
            turns = extract_conversation(conv)
            for turn in turns:
                if "AI" in turn[0]:
                    gpt_turns += 1
        else:
            gpt_turns += len(entry.get("dict_list", []))

    print(f"Total GPT turns: {gpt_turns}")
    print("COMPLETED FINAL PHASE")
    if USE_SUBSET:
        print(f"Warning! USE_SUBSET was on in the config you used, {config_path}. This means that you only generated data from the first {SUBSET_SIZE} chunks of your input data. If you want to generate data from all chunks, set USE_SUBSET to False.")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
