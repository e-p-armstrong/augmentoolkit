import asyncio
import glob
import os
import random
import sys
import traceback

import yaml

from augmentoolkit.classifier_creator.steps import all_labels_same, create_label, create_rules, run_classifier, save_train_set, train_classifier
from augmentoolkit.control_flow_functions import control_flow_functions
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper

# will go in eventual utils
# Each pipeline should have its own utils
def sample_and_remove(lst, n):
    sampled = []
    for _ in range(min(n, len(lst))):
        index = random.randrange(len(lst))
        sampled.append(lst.pop(index))
    return sampled

async def main():
    
    with open("./config_classifier.yaml", "r") as f: # different yaml file for different pipes
        config = yaml.safe_load(f)
        
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
    
    
    extensions = [".txt", ".md"]

    source_texts = []
    for extension in extensions:
      path = f"{INPUT_FOLDER}/**/*" + extension
      source_texts = source_texts + glob.glob(path, recursive=True)
      
    chunks = []
    for source_text in source_texts:
        chunks += control_flow_functions.sentence_chunking_algorithm(
            source_text, config["SYSTEM"]["CHUNK_SIZE"]
        )
        
    if TRAIN_SET_SIZE + TEST_SET_SIZE > len(chunks):
        print("\n\nTRAIN SET SIZE AND TEST SET SIZE TOO LARGE FOR EVEN A SINGLE CLASSIFIER TRAINING RUN GIVEN THE SIZE OF THE DATASET")
        print("REDUCE TRAIN OR TEST SET SIZE, OR ADD MORE INPUT DATA")
        print(f"For reference, the total length of the chunks is {len(chunks)}")
        sys.exit(1)
        
    conversions = [("\n", " "), ("  ", " ")]

    chunks = [
        (control_flow_functions.fix_text(conversions, seq[0]), seq[1])
        for seq in chunks
    ]
    print("Chunking succeeded")
        
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
    
    rules_string = await create_rules(engine_wrapper=engine_wrapper_large, classes_list=USER_CLASSES, classes_desc=USER_CLASSES_DESCRIPTION, completion_mode=COMPLETION_MODE)
    print("Rules created!\n\n----------------")
    print(rules_string)
    print("-------------")
    
    # Sample 1000 things from the input dataset at random and classify them (plus a test dataset of 50)
    train_data = sample_and_remove(chunks, TRAIN_SET_SIZE)
    print("Training data sampled")
    print(f"Length of training data: {len(train_data)}")

    # Create directory if it doesn't exist
    output_dir = os.path.join(config["PATH"]["OUTPUT"], "text_label_tuples")
    os.makedirs(output_dir, exist_ok=True)
    
    text_label_tuples = []
    
    ## Create initial training data using the small LLM, make text-label pairs
    await run_async_many(engine_wrapper=engine_wrapper, output_dir=output_dir, input_list=train_data, func=create_label, output_list=text_label_tuples,rules=rules_string) # TODO need to add to this the actual label list and desc somehow
    
    # TODO remove breakpoint below
    input("\n\nHIT ENTER TO CONTINUE AFTER MANUALLY MAKING THE DATA WORK")
    ###
    
    classifier_counter = 0 # incremented whenever a new classifier is made
    output_dir = os.path.join(config["PATH"]["OUTPUT"], "classifiers")
    os.makedirs(output_dir, exist_ok=True)
    model = train_classifier(text_label_tuples, classifier_counter, output_dir)
    
    ### Test classifier against LLM
    
    has_passed_LLM_validation = False
    
    while not has_passed_LLM_validation:
        
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
            run_async_many(engine_wrapper=engine_wrapper_large, output_dir=output_dir, input_list=test_set, func=create_label, output_list=truth_labels,rules=rules_string) # the create_label function should have validation built in, maybe # TODO need to add to this the actual label list and desc somehow
            
            output_dir = os.path.join(config["PATH"]["OUTPUT"], "classifier_testing_labels_classification")
            os.makedirs(output_dir, exist_ok=True)
            
            classifier_labels = []
            run_async_many(model, output_dir, input_list=test_set, func=run_classifier, output_list=classifier_labels) # TODO need to add to this the actual label list and desc somehow
            
            # Compare the two
            if len(truth_labels) != len(classifier_labels):
                print("\n\nLIST LENGTHS NOT EQUIVALENT")
                pass # If this is true, something is broken
            elif all_labels_same(truth_labels, classifier_labels): # all_labels_same will have to work regardless of item order, since async. Also, most control_flow_functions. will actually end up being pipeline-specific functions instead.
                has_passed_LLM_validation = True
            else:
                text_label_tuples += truth_labels
                
                output_dir = os.path.join(config["PATH"]["OUTPUT"], "text_label_tuples")
                os.makedirs(output_dir, exist_ok=True)
                new_train_samples_inputs = sample_and_remove(chunks, TRAIN_SET_INCREMENT)
                new_train_samples = []
                run_async_many(new_train_samples_inputs, engine_wrapper, output_dir, input_list=new_train_samples_inputs, func=create_label, output_list=new_train_samples,rules=rules_string)
                
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
    print("Executing on entire set...")
    
    output_dir = os.path.join(config["PATH"]["OUTPUT"], "final_classifier_output")
    os.makedirs(output_dir, exist_ok=True)
    run_async_many(classifier_labels, model, output_dir, input_list=chunks, func=run_classifier, output_list=classifier_labels)
    
asyncio.run(main())