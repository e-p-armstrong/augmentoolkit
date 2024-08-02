import json
import logging
import os
import re
import traceback
from augmentoolkit.generation_functions.generation_step_class import GenerationStep
from augmentoolkit.utils.make_id import make_id
from augmentoolkit.utils.write_output_to_file import write_output_to_file


class PipelineStep:
    def __init__(
        self, 
        prompt_path=None,
        default_prompt_folder=None,
        sampling_params=None,
        output_dir=None,
        output_subdir=None,
        save_path=None,
        output_processor=None,
        completion_mode=False,
        use_stop=True,
        logging_level=logging.INFO,
        prompt_folder=None,
        intermediate_output_path=None,
        result_key="placeholder_result_key", # this is the key that the result will be saved under in the output dictionary.
        regex=re.compile(r".*", re.DOTALL),
        
        ): # things that are args here are things that would be in the code. Some of these will be live-tweakable.
        self.prompt_path = prompt_path + ".yaml" if not completion_mode else prompt_path + ".txt"
        self.sampling_params = sampling_params
        self.output_dir = output_dir
        self.save_path = save_path
        self.output_processor = output_processor
        self.completion_mode = completion_mode
        self.default_prompt_folder = default_prompt_folder
        self.logging_level = logging_level
        self.use_stop = use_stop
        self.prompt_folder = prompt_folder
        self.intermediate_output_path = intermediate_output_path
        self.result_key = result_key
        self.regex = regex
        self.output_subdir = output_subdir
        self.full_output_path = os.path.join(self.output_dir, self. output_subdir)
        self.intermediate_output_path_full = os.path.join(self.full_output_path, self.intermediate_output_path)
    
    def process_input_data(self, input_data):
        return input_data # this should be a dictionary with the keys being the same as the interpolation spots in the prompt. This function in particular will basically always be overridden in subclasses.
    
    def make_save_path_file(self, idx):
        return os.path.join(self.full_output_path, self.save_path, f"{str(idx)}.json")
    
    def read_previous_output(self, idx, output_list):
        save_path_file = self.make_save_path_file(idx)
        if os.path.exists(save_path_file):
            with open(save_path_file, "r") as f:
                output_data = json.load(f)
            output_list.append(output_data)
            return True
        return False

    
    async def generate_data(self, processed_data, engine_wrapper):
        try:
                
            generator = GenerationStep(
                prompt_path=self.prompt_path,
                default_prompt_folder=self.default_prompt_folder,
                sampling_params=self.sampling_params,
                completion_mode=self.completion_mode,
                engine_wrapper=engine_wrapper,
                output_processor=self.output_processor,
                retries=3, 
                logging_level=self.logging_level,
                use_stop=self.use_stop,
                prompt_folder=self.prompt_folder,
                regex=self.regex,
            )
            
            print("DEBUG PROCESSED DATA")
            print(processed_data)
            
            result, full_output = await generator.generate(**processed_data)
            
            return result, full_output
        except Exception as e:
            print(e)
            traceback.print_exc()
    
    
    
    def save(self, result=None,
    full_output=None,
    idx=None,
    output_list=None,
    input_data=None,):
        save_path_file = self.make_save_path_file(idx)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_data = input_data
        output_data[self.result_key] = result
        write_output_to_file(full_output, self.intermediate_output_path_full, idx)
        
        os.makedirs(self.save_path, exist_ok=True)
        with open(save_path_file, "w") as f:
            f.write(json.dumps(output_data))
        
        output_list.append(output_data)
    
    async def run(self, idx=None,
    input_data=None,
    engine_wrapper=None,
    output_list=None,
      ): # things that are args here are produced during inference time. Including config settings.
        
        read_previous_item = self.read_previous_output(idx, output_list)
        if read_previous_item:
            return
        
        processed_data = self.process_input_data(input_data)
        
        result, full_output = await self.generate_data(processed_data, engine_wrapper)
        
        self.save(result=result, full_output=full_output, idx=idx, output_list=output_list, input_data=input_data)
        

        