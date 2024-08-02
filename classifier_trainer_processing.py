import asyncio

from augmentoolkit.utils.head_tail_truncate import head_tail_truncate
from augmentoolkit.utils.load_dataset import load_dataset
import augmentoolkit.utils.sentence_chunking_algorithm

async def main():
    
    print("You may occasionally see exceptions thrown if/when the LLM messes up, but if the pipeline keeps running then these have been caught and regenerated and everything is chugging along fine. Keep Calm and Carry On. The reason the exceptions are so big is because that gives more information for potentially debugging prompts.")
    print("Happy dataset generation and classifier model creation!")
    import yaml
    import glob
    import json
    import os
    import random
    import sys
    import traceback

    from augmentoolkit.utils.sample_and_remove import sample_and_remove

    from augmentoolkit.classifier_trainer.steps import all_labels_same, create_label, create_rules, run_classifier, save_train_set, train_classifier
    from augmentoolkit.core import steps
    from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper

    with open("./classifier_trainer_config.yaml", "r") as f: # different yaml file for different pipes
        config = yaml.safe_load(f)
    random.seed(1048596)
        
    if not os.path.exists(config["PATH"]["OUTPUT"]):
        os.makedirs(config["PATH"]["OUTPUT"])
        
    LOGICAL_MODEL = config["API"]["LOGICAL_MODEL"]

    LARGE_LOGICAL_MODEL = config["API"]["LARGE_LOGICAL_MODEL"]
 
    DOUBLE_CHECK_COUNTER = config["SYSTEM"][
        "DOUBLE_CHECK_COUNTER"
    ]  # Set to 1 to check outputs only once; set to 2 to check twice; set to 3 to check thrice, etc. Set to 0 to break everything in vet_question_loop() and elsewhere. Set to -1 and cause the universe to implode?

    CONCURRENCY_LIMIT = config["SYSTEM"][
        "CONCURRENCY_LIMIT"
    ]  # Adjust this number based on the rate limit constraints of your api

    API_KEY = config["API"]["API_KEY"]

    BASE_URL = config["API"][
        "BASE_URL"
    ]  # Augmentoolkit-API should also be compatible with any other API provider that accepts OAI-style requests

    COMPLETION_MODE = config["SYSTEM"]["COMPLETION_MODE"]

    MODE = config["SYSTEM"]["MODE"]

    INPUT_FOLDER = config["PATH"]["INPUT"]
    
    USER_CLASSES = config["CLASSIFICATION"]["CLASSES"] # Something like ["happy", "sad", "angry"] or ["great", "bad"] or ["mature", "safe"] --- a list of classes
    USER_CLASSES_DESCRIPTION = config["CLASSIFICATION"]["DESC"] # A description of the classes. "Classify text based on its emotional content and vibe, such as happy, sad, or angry" or "I need text to be classified based on whether it's high-quality (great) or lame (bad)" or "Classify the text based on whether it contains mature content or not"
    
    TRAIN_SET_SIZE = config["TRAINING"]["TRAIN_SET_SIZE"]
    TRAIN_SET_INCREMENT = config["TRAINING"]["TRAIN_SET_INCREMENT"]
    TEST_SET_SIZE = config["TRAINING"]["TEST_SET_SIZE"]
    REQUIRED_ACCURACY = config["SYSTEM"]["REQUIRED_ACCURACY"]
    CHUNK_SIZE = config["SYSTEM"]["CHUNK_SIZE"]
    PREDICT_ON_WHOLE_SET_AT_THE_END = config["CLASSIFICATION"]["PREDICT_ON_WHOLE_SET_AT_THE_END"]
    TRUNCATION_TYPE = config["TRAINING"]["TRUNCATION_TYPE"]
    
    extensions = [".txt", ".md", ".json", ".jsonl", ".parquet"]
    
    source_texts = []
    for extension in extensions:
        path = f"{INPUT_FOLDER}/**/*{extension}"
        source_texts.extend(glob.glob(path, recursive=True))

    chunks = []
    for source_text in source_texts:
        if source_text.endswith(('.txt', '.md')):
            chunks.extend(augmentoolkit.utils.sentence_chunking_algorithm.sentence_chunking_algorithm(
                source_text, CHUNK_SIZE
            ))
        elif source_text.endswith(('.json', '.jsonl', '.parquet')):
            dataset = load_dataset(source_text)
            if 'text' not in dataset.columns:
                print(f"Warning: 'text' column not found in {source_text}. Skipping this file.")
                continue
            for text in dataset['text']:
                if TRUNCATION_TYPE == "head-tail":
                    truncated_text = head_tail_truncate(text, max_length=CHUNK_SIZE)
                else:
                    truncated_text = text[:CHUNK_SIZE]
                chunks.append((truncated_text, source_text))

    if TRAIN_SET_SIZE + TEST_SET_SIZE > len(chunks):
        print("\n\nTRAIN SET SIZE AND TEST SET SIZE TOO LARGE FOR EVEN A SINGLE CLASSIFIER TRAINING RUN GIVEN THE SIZE OF THE DATASET")
        print("REDUCE TRAIN OR TEST SET SIZE, OR ADD MORE INPUT DATA")
        print(f"For reference, the total length of the chunks is {len(chunks)}")
        sys.exit(1)

    conversions = [("\n", " "), ("  ", " ")]

    chunks = [
        (steps.fix_text(conversions, seq[0]), seq[1])
        for seq in chunks
    ]
    random.shuffle(chunks)
    print("Chunking succeeded")
    print("-----------------\nExample chunks:")
    print(chunks[0])
    print("-----------------")
        
    from tqdm import asyncio as tqdmasyncio
    import asyncio

    # Set up rate-limit-conscious functions
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def run_task_with_limit(task):
        async with semaphore:
            # Run your task here
            return await task
        
    async def run_async_many(*args, input_list=None, func=None, **kwargs):
        tasks = [
        func(
            idx,
            inp,
            *args,
            **kwargs,
            ) for idx, inp in enumerate(input_list)
        ]

        task_list = [run_task_with_limit(task) for task in tasks]
        for future in tqdmasyncio.tqdm.as_completed(task_list):
            await future

    engine_wrapper = EngineWrapper(
        model=LOGICAL_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        mode=MODE,
        # quantization="gptq" # modify if you want to do stuff with the aphrodite branch
    )
    
    engine_wrapper_large = EngineWrapper(
        model=LARGE_LOGICAL_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        mode=MODE,
        # quantization="gptq" # modify if you want to do stuff with the aphrodite branch
    )
    
    # First, create the 5 rules for classifying text based on the classes and desc
    
    # Load rules if present, otherwise create them
    
    import os
    import yaml

    if os.path.exists(os.path.join(config["PATH"]["OUTPUT"], "rules_creation_generation")):
        yaml_files = [f for f in os.listdir(os.path.join(config["PATH"]["OUTPUT"], "rules_creation_generation")) if f.endswith('.yaml')]
        if yaml_files:
            yaml_file_path = os.path.join(config["PATH"]["OUTPUT"], "rules_creation_generation", yaml_files[0])
            with open(yaml_file_path, 'r') as file:
                yaml_content = yaml.safe_load(file)
                if isinstance(yaml_content, list) and yaml_content:
                    print("Loading preexisting rules...")
                    rules_string = yaml_content[-1]['content']
                else:
                    rules_string = await create_rules(engine_wrapper=engine_wrapper_large, classes_list=USER_CLASSES, classes_desc=USER_CLASSES_DESCRIPTION, completion_mode=COMPLETION_MODE)
        else:
            rules_string = await create_rules(engine_wrapper=engine_wrapper_large, classes_list=USER_CLASSES, classes_desc=USER_CLASSES_DESCRIPTION, completion_mode=COMPLETION_MODE)
    else:
        rules_string = await create_rules(engine_wrapper=engine_wrapper_large, classes_list=USER_CLASSES, classes_desc=USER_CLASSES_DESCRIPTION, completion_mode=COMPLETION_MODE)

    print("Rules created!\n\n----------------")
    print(rules_string)
    print("-------------")
    
    output_dir = os.path.join(config["PATH"]["OUTPUT"], "text_label_tuples")
    saved_tuples_dir = os.path.join(output_dir, "saved_label_tuples")

    text_label_tuples = []

    # Load existing tuples if they exist
    if os.path.exists(saved_tuples_dir):
        json_files = glob.glob(os.path.join(saved_tuples_dir, "*.json"))
        for file in json_files:
            with open(file, 'r') as f:
                tuple_data = json.load(f)
                if isinstance(tuple_data, list) and len(tuple_data) == 3:
                    text_label_tuples.append(tuple_data)

    # Determine how many more tuples we need to generate
    remaining_tuples = max(0, TRAIN_SET_SIZE - len(text_label_tuples))

    # Sample and remove from chunks if needed
    train_data = []
    if remaining_tuples > 0:
        train_data = sample_and_remove(chunks, remaining_tuples)

    print("Training data prepared")
    print(f"Loaded tuples: {len(text_label_tuples)}")
    print(f"Tuples to generate: {len(train_data)}")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate remaining tuples if needed
    if train_data:
        await run_async_many(engine_wrapper=engine_wrapper, output_dir=output_dir, input_list=train_data, func=create_label, output_list=text_label_tuples, rules=rules_string, classes=USER_CLASSES)
    
    with open(os.path.join(config["PATH"]["OUTPUT"], "TEST_DEBUG_OUTPUT_OF_LIST"),  'w') as f:
        f.write(json.dumps(text_label_tuples))
    
    classifier_counter = 0
    output_dir = os.path.join(config["PATH"]["OUTPUT"], "classifiers")
    os.makedirs(output_dir, exist_ok=True)

    # Count existing classifier folders
    existing_classifiers = glob.glob(os.path.join(output_dir, "classifier_*"))
    classifier_counter = len(existing_classifiers)

    model = train_classifier(text_label_tuples, classifier_counter, output_dir)
    
    ### Test classifier against LLM
    
    has_passed_LLM_validation = False
    max_iters = config["TRAINING"]["MAX_ITERS"]
    
    while not has_passed_LLM_validation and max_iters > 0:
        max_iters = max_iters - 1
        if chunks: # if we still have content; else, if it's empty, the classifier is as good as we'll get and we exit early
            # make the output dir
            output_dir = os.path.join(config["PATH"]["OUTPUT"], "truth_labels_classification")
            os.makedirs(output_dir, exist_ok=True)
            
            # First, take out a test set
            test_set = sample_and_remove(chunks, TEST_SET_SIZE)
            
            # filter out duplicates
            test_set = [item for idx, item in enumerate(test_set) if len([i for i in test_set[idx:] if i[0] == item[0]]) == 1]
            
            truth_labels = []
            
            # Do LLM testing on that test set
            await run_async_many(engine_wrapper=engine_wrapper_large, output_dir=output_dir, input_list=test_set, func=create_label, output_list=truth_labels,rules=rules_string, classes=USER_CLASSES) # the create_label function should have validation built in, maybe # TODO need to add to this the actual label list and desc somehow
            
            output_dir = os.path.join(config["PATH"]["OUTPUT"], "classifier_testing_labels_classification")
            os.makedirs(output_dir, exist_ok=True)
            
            classifier_labels = []
            run_classifier(model=model, output_dir=output_dir, input_list=test_set, output_list=classifier_labels)
            # run_async_many(model, output_dir, input_list=test_set, func=run_classifier, output_list=classifier_labels) # TODO need to add to this the actual label list and desc somehow
            
            # Compare the two
            if len(truth_labels) != len(classifier_labels):
                print("\n\nLIST LENGTHS NOT EQUIVALENT")
                print(f"len(truth_labels) {len(truth_labels)} vs len(classifier_labels) {len(classifier_labels)}")
                pass # If this is true, something is broken
            elif all_labels_same(truth_labels, classifier_labels, required_accuracy=REQUIRED_ACCURACY): # all_labels_same will have to work regardless of item order, since async. Also, most control_flow_functions. will actually end up being pipeline-specific functions instead.
                has_passed_LLM_validation = True
            else:
                text_label_tuples += truth_labels
                
                output_dir = os.path.join(config["PATH"]["OUTPUT"], "text_label_tuples")
                os.makedirs(output_dir, exist_ok=True)
                new_train_samples_inputs = sample_and_remove(chunks, TRAIN_SET_INCREMENT)
                new_train_samples = []
                await run_async_many(engine_wrapper=engine_wrapper, output_dir=output_dir, input_list=new_train_samples_inputs, func=create_label, output_list=new_train_samples, rules=rules_string, classes=USER_CLASSES)
                
                text_label_tuples += new_train_samples
                
                
                output_dir = os.path.join(config["PATH"]["OUTPUT"], "classifier_training_set")
                save_train_set(text_label_tuples, output_dir)
                
                output_dir = os.path.join(config["PATH"]["OUTPUT"], "classifiers")
                classifier_counter += 1
                model = train_classifier(text_label_tuples, classifier_counter, output_dir)
        else:
            print("Ran out of training chunks")
            sys.exit(1) # TODO failure logic
    
    print("finished training classifier")
    print(f"ITERATION COMPLETE\nITERATIONS DONE: {max_iters}\nDID REACH THRESHOLD?: {has_passed_LLM_validation}")

    if PREDICT_ON_WHOLE_SET_AT_THE_END:
        print("Executing on entire set...")
        
        output_dir = os.path.join(config["PATH"]["OUTPUT"], "final_classifier_output")
        os.makedirs(output_dir, exist_ok=True)
        run_classifier(model=model, output_dir=output_dir, input_list=chunks, output_list=classifier_labels)
    # run_async_many(classifier_labels, model, output_dir, input_list=chunks, func=run_classifier, output_list=classifier_labels)
    
asyncio.run(main())