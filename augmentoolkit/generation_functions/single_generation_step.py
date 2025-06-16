import hashlib
import json
import os
import traceback
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep

# NOTE (performance fix) this stays as it is because it's always just one file.


class SingleGenerationStep(PipelineStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_hash(self, input_data):
        """Generate a unique hash from the input data to use as filename"""
        # Convert input data to a stable string representation
        input_str = json.dumps(input_data, sort_keys=True)
        # Create hash
        return hashlib.md5(input_str.encode()).hexdigest()

    def make_output_path(self, output_dir, input_hash):
        """Returns full path to the output text file based on input hash"""
        return os.path.join(output_dir, f"{self.output_file}_{input_hash}.txt")

    async def read_previous_output(self, input_data, output_dir):
        """Check if output for this input already exists"""
        input_hash = self.generate_hash(input_data)
        output_path = self.make_output_path(output_dir, input_hash)

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save(
        self,
        result=None,
        input_data=None,
        output_dir=None,
        full_output=None,
        full_response=None,
        full_input=None,
        include_details=False,
        completion_mode=False,
    ):
        """Save result to a text file named by input hash"""
        input_hash = self.generate_hash(input_data)
        output_path = self.make_output_path(output_dir, input_hash)

        # Ensure directory exists before any file operations
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if self.log_full_outputs and full_output:
            print("\n" + "=" * 50)
            print(f"FULL OUTPUT FOR HASH {input_hash}:")
            print("=" * 50)

            # If full_output is a dictionary, print it with nice formatting
            if isinstance(full_output, dict):
                for key, value in full_output.items():
                    print(f"\n{key.upper()}:")
                    print("-" * 50)
                    print(value)
            else:
                # Otherwise just print the full output directly
                print(full_output)

            print("=" * 50 + "\n")

        if include_details:
            obj = {
                self.result_key: result,
                "full_output": full_output,
                "full_response": full_response,
                "full_input": full_input,
                "completion_mode": completion_mode,
            }
        else:
            obj = {self.result_key: result}

        # Use file lock for safe concurrent writes
        # Write to temporary file first then replace for atomic write
        temp_path = output_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(temp_path, output_path)

        return obj

    async def run(
        self,
        input_data=None,
        engine_wrapper=None,
        default_prompt_folder=None,
        prompt_folder=None,
        output_dir=None,
        completion_mode=False,
        use_stop=True,
        include_details=False,
        **kwargs,
    ):
        full_prompt_path = (
            self.prompt_path + ".yaml"
            if not completion_mode
            else self.prompt_path + ".txt"
        )

        # Check if this input has already been processed
        previous_result = await self.read_previous_output(input_data, output_dir)
        if previous_result is not None:
            return previous_result

        processed_data, additional_kwargs = self.process_input_data(input_data)

        error_message = ""
        complete = False
        max_retries = self.max_retries
        while not complete and max_retries > 0:
            try:
                result, full_output, full_response, full_input = (
                    await self.generate_data(
                        processed_data,
                        engine_wrapper,
                        full_prompt_path,
                        prompt_folder,
                        default_prompt_folder,
                        completion_mode,
                        use_stop,
                        error_message=error_message,
                        **kwargs,
                        **additional_kwargs,
                    )
                )

                validation_result = self.validation_function(result, input_data)
                if validation_result["result"]:
                    complete = True
                else:
                    error_message = validation_result["message"]
            except Exception as e:
                print(e)
                error_message = str(e)
                traceback.print_exc()
            max_retries -= 1

        if not complete:
            return None

        return self.save(
            result=result,
            input_data=input_data,
            output_dir=output_dir,
            full_output=full_output,
            full_response=full_response,
            full_input=full_input,
            include_details=include_details,
            completion_mode=completion_mode,
        )
