import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
from datasets import load_dataset
from typing import Optional
import re
from utils import calculate_rep
# from accelerate import Accelerator
# from peft import (
#     LoraConfig,
#     PeftConfig,
#     PeftModel,
#     get_peft_model,
#     prepare_model_for_kbit_training
# )
# from transformers import (
#     AutoConfig,
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig
# )
# from transformers import pipeline
import numpy as np
from transformers import AutoTokenizer, pipeline
from ragLLM import RAG

class LLM:  
    def __init__(self) -> None:
        self.key = '' 
        self.device ='cuda:1'
        self.model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
        # self.device = 'auto'
        # Use a pipeline as a high-level helper
        self._pipeline = pipeline(
            task="text-generation",
            model=self.model_name,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            }, 
            device_map=self.device        
        )
        self.prompt_history = []
        self.message_history = []
        self.llm_eval_score = 5
        self.numerator = 1
        self.denominator = 2
        self.count = 1

    #imporve the follow_up llm no links to RAG knowledge.
    def follow_up_llm(self, user_score:int, prompt: str):
        rep, numerator, denominator = calculate_rep(user_score, self.llm_eval_score, self.count, self.numerator, self.denominator)
        self.numerator = numerator
        self.denominator = denominator
        print(rep)
        if rep < 0.5:
            messages = [{"role": "system", "content": "You are to ask follow up questions to help formulate a better response"},
                        {"role": "user", "content": prompt },]
            self.message_history = self.message_history + messages
            prompt = self._pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
            
            terminators = [
                self._pipeline.tokenizer.eos_token_id,
                self._pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = self._pipeline(
                prompt,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            return outputs[0]["generated_text"][len(prompt):]
        else: 
            return "Thank you for your feedback"
    
    def break_tasks(self, prompt, num_task = 2) -> list[str]:
        messages = [{"role": "system", "content": f"You are the main task coordinator, break the complex prompt into at most {num_task} easier prompts, such that smaller models can complete it. Add a # after every sub-task"},
                        {"role": "user", "content": prompt },]
        prompt = self._pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True)
                
        terminators = [
            self._pipeline.tokenizer.eos_token_id,
            self._pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self._pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        generated_text =  outputs[0]["generated_text"][len(prompt):]
        # Split the string based on the character "#"
        split_string = generated_text.strip().split("#")
        # Remove any empty strings and leading/trailing spaces
        prompt_array = [s for s in split_string if s]
        output_array = []
        for idx, mini_prompts in enumerate(prompt_array):
            if idx == 0:
                continue
            print('----sub-prompts---')
            print(mini_prompts)
            print('----OutPUT:---')
            
            output = self.infer_llm(str(mini_prompts), is_complex=False)
            print(output)

            output_array.append(output)

        combined_answer = ' '.join(output_array)
        return str(combined_answer)

    #need to extract the score and pass it to reputation calculator
    def evaluate_llm_resp(self, prompt:str):
        messages = [{"role": "system", 
                     "content": "You are an evaluator for the following response from a LLM, Please provide:\n"
        "- A score between 0 to 10 evaluating on completion, quality, and overall performance"
        "Check if the response have any hallucinations"
        "- Bullet points suggestions to improve similar tasks in the future \n" 
        },
        {"role": "user",
          "content": prompt},]
        prompt = self._pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        
        terminators = [
            self._pipeline.tokenizer.eos_token_id,
            self._pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self._pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        # print(outputs[0])
        self.llm_eval_score = self.find_score(outputs[0]["generated_text"][len(prompt):])
        return (outputs[0]["generated_text"][len(prompt):])

    def rag_llm(self, prompt:str):
        rag_use = RAG()
        rag_use.read_docs(['/home/xavier002/Private_LLM/data/activelearning.pdf'])
        rag_use.initiate_models()
        response = rag_use.rag_query(prompt)
        return response    
    
    def infer_llm(self, prompt , is_complex=True):
        rag_response = None
        self.count +=1
        generated_response = None
        # print(prompt)
        if is_complex:
            generated_response = str(self.break_tasks(prompt))
        if prompt:
            check_rag = re.search(r"\bRAG\b", prompt)
            if check_rag:
                print('-----using RAG----')
                rag_response = self.rag_llm(prompt)
                return str(rag_response)
                
            messages = [{"role": "system", "content": "You are a helpful assistant and answer the prompts to the best of your ability. "},
                        {"role": "user", "content": prompt  },]
            self.prompt_history.append(prompt)
            max_window = max(3, len(self.prompt_history))
            self.prompt_history[:max_window]
            self.message_history = self.message_history + messages
        else:
            messages = [{"role": "system", "content": "You are a helpful assistant!"},
                        {"role": "user", "content": "What model are you based on" },]
        

        prompt = self._pipeline.tokenizer.apply_chat_template(
        self.message_history ,
        tokenize=False,
        add_generation_prompt=True)

        terminators = [
            self._pipeline.tokenizer.eos_token_id,
            self._pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self._pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
            
        if generated_response == None:
            generated_response = outputs[0]["generated_text"][len(prompt):]

        return generated_response

    def get_pipeline(self):
        return self._pipeline

    def find_score(self, response):
        score = re.search("[1]?[0-9]", response)
        if score:
            print(score.group(0))
            return int(score.group(0))
        else:
            return 3

