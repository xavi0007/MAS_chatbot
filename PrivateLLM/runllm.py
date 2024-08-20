import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
from datasets import load_dataset
from typing import Optional
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

class LLM:  
    def __init__(self) -> None:
        self.key = '' 
        self.device ='cuda:0'
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
    def follow_up_llm(self, rep: float, prompt: str):
        if rep < 0.5:
            messages = [{"role": "system", "content": "You are to be a more helpful assistant and should ask follow up questions to help formulate a better response"},
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
        else: return "Thank you for your feedback"
    
    
      
    def infer_llm(self, prompt:str, roleplay: Optional[str]):
        
        if prompt and roleplay=='user':
            messages = [{"role": "system", "content": "You are a helpful assistant and answer the prompts to the best of your ability. "},
                        {"role": "user", "content": prompt  },]
            self.prompt_history.append(prompt)
            max_window = max(3, len(self.prompt_history))
            self.prompt_history[:max_window]
            self.message_history = self.message_history + messages
            # self.chat_history.append(messages)
        else:
            messages = [{"role": "system", "content": "You are a helpful assistant!"},
                        {"role": "user", "content": "What model are you based on" },]
            
  
        prompt = self._pipeline.tokenizer.apply_chat_template(
        self.message_history ,
        tokenize=False,
        add_generation_prompt=True)

        if roleplay == 'evaluator':
            #prompt is the response in this case
            messages = [{"role": "system", "content": "You are an evaluator for the following response, Please provide:\n"
            f"- Bullet points suggestions to improve future similar tasks and prompt {self.prompt_history[-1]} \n" 
            "Check if the response have any hallucinations"
            "- A score from 0 to 10 evaluating on completion, quality, and overall performance"},
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
        # print(outputs[0]["generated_text"][len(prompt):])
        return outputs[0]["generated_text"][len(prompt):]

    def get_pipeline(self):
        return self._pipeline

# llm = LLM()
# llm.infer_llm("test")
