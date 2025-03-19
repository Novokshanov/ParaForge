#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from typing import List, Dict, Any, Optional
import sys
from tqdm import tqdm

def initialize_model() -> Any:
    """
    Initialize the language model using vLLM if available, or fall back to transformers.
        
    Returns:
        Initialized model
    """
    # Check if GPU is available
    import torch
    gpu_available = torch.cuda.is_available()
    
    # Determine device to use
    if gpu_available:
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")
    
    model_name: str = "Seelt/Qwen2.5-0.5B-ParaForge-v0.1"
    
    # Try to use vLLM first
    try:
        from vllm import LLM, SamplingParams
        
        # Set tensor parallel size to 1 for single GPU or CPU
        tensor_parallel_size = 1
        
        # Initialize the model with vLLM
        model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,  # Required for some models like Qwen
            dtype="float16" if device == "cuda" else "float32",  # Use half precision for GPU
        )
        
        print(f"Successfully initialized model with vLLM: {model_name}")
        return model
        
    except (ImportError, ModuleNotFoundError) as e:
        print(f"vLLM not properly installed or not compatible with your system: {str(e)}")
        print("Falling back to transformers library...")
        
        try:
            # Fall back to transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            
            if device == "cpu":
                model = model.to("cpu")
            
            # Create a wrapper object to maintain compatibility with the rest of the code
            class TransformersWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.device = device
                
                def generate(self, prompts, sampling_params):
                    results = []
                    
                    for prompt in tqdm(prompts, desc="Generating translations"):
                        # prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                        inputs = self.tokenizer(prompt, return_tensors="pt")
                        if self.device == "cuda":
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            output_ids = self.model.generate(
                                inputs["input_ids"],
                                max_new_tokens=24,
                                do_sample=True,
                                # temperature=0.9,
                                top_p=0.9,
                                num_return_sequences=1,
                                pad_token_id=self.tokenizer.eos_token_id,
                            )
                        
                        # Decode
                        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        # Extract the generated part (after the prompt)
                        generated_text = output_text[len(prompt):]
                        generated_text = generated_text.split("\n")[0]
                        # generated_text = output_text
                        
                        # Create a result object similar to vLLM's output
                        class Output:
                            def __init__(self, text):
                                self.text = text
                        
                        class OutputWrapper:
                            def __init__(self, text):
                                self.outputs = [Output(text)]
                        
                        results.append(OutputWrapper(generated_text))
                    
                    return results
            
            wrapper = TransformersWrapper(model, tokenizer)
            print(f"Successfully initialized model with transformers: {model_name}")
            return wrapper
            
        except Exception as e:
            print(f"Error initializing model with transformers: {str(e)}")
            print("Could not initialize any model. Please check your installation.")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error initializing model {model_name}: {str(e)}")
        raise

def generate_parallel_sentences(model, sentences: List[str]) -> List[str]:
    """
    Generate parallel sentences using the language model.
    
    Args:
        model: Initialized model (either vLLM or transformers wrapper)
        sentences: List of original sentences
        
    Returns:
        List of generated sentences
    """

    tag_instructions = """<тэги>
    <лицо>
    1 - первое лицо;
    2 - второе лицо;
    3 - третье лицо;
    </лицо>
    <род>
    masc - мужской;
    fem - женский;
    </род>
    <число>
    sg - единственное число;
    pl - множественное число;
    </число>
    <падежи>
    nom - именительный;
    gen - родительный;
    dat - дательный;
    acc - винительный;
    abl - творительный;
    loc - предложеный;
    </падежи>
    <время>
    pst - прошедшее;
    prs - настоящее;
    fut - будущее;
    </время>
    </тэги>"""

    instruction_template = f"""
    Инструкция: Восстанови предложение из лемм и грамматических тегов.
    Описание тегов:
    {tag_instructions}

    Пример:
    Input: мама<sg,nom> драить[мыть,чистить]<3,fem,sg,pst> грязный окно<sg,acc>
    Output: Мама драила грязное окно

    Теперь заполни:
    Input: {{input}}
    Output:
    """
    if not sentences:
        return []
    
    # Import SamplingParams if it's not already imported
    try:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.9,  # Match transformers temperature
            top_p=0.9,       # Match transformers top_p
            max_tokens=24,   # Match transformers max_new_tokens
            n=1,            # Equivalent to num_return_sequences=1
        )
    except:
        # Create a simple class to mimic SamplingParams for the transformers fallback
        class SamplingParamsSimple:
            def __init__(self, temperature=0.9, top_p=0.9, max_tokens=24, do_sample=True):
                self.temperature = temperature
                self.top_p = top_p
                self.max_tokens = max_tokens
        
        sampling_params = SamplingParamsSimple(
            temperature=0.9,
            top_p=0.9,
            max_tokens=24,
        )
    
    # Prepare prompts for each sentence
    prompts = []
    for sentence in sentences:
        # Create a simple prompt that asks the model to rephrase the sentence
        prompts.append(instruction_template.format(input=sentence))
    
    # Generate responses in batch for efficiency
    try:
        outputs = model.generate(prompts, sampling_params)
        
        # Extract generated sentences from outputs
        generated_sentences = []
        for output in outputs:
            # Get the generated text and clean it up
            generated_text = output.outputs[0].text.strip()
            
            # Remove quotes if present
            if generated_text.startswith('"') and generated_text.endswith('"'):
                generated_text = generated_text[1:-1]
                
            generated_sentences.append(generated_text)
        
        return generated_sentences
        
    except Exception as e:
        print(f"Error generating sentences: {str(e)}")
        # Return empty strings in case of error
        return [""] * len(sentences) 