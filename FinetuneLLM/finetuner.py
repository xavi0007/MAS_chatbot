import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig
from trl import SFTTrainer
import sys
from subprocess import call

print('_____Python, Pytorch, Cuda info____')
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA RUNTIME API VERSION')
#os.system('nvcc --version')
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('_____nvidia-smi GPU details____')
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('_____Device assignments____')
print('Number CUDA Devices:', torch.cuda.device_count())
print ('Current cuda device: ', torch.cuda.current_device(), ' **May not correspond to nvidia-smi ID above, check visibility parameter')
print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

device ='auto'
#################################
### Setup Quantization Config ###
#################################
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)


#######################
### Load Base Model ###
#######################
base_model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
llama_3 = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map=device
)

######################
### Load Tokenizer ###
######################
tokenizer = AutoTokenizer.from_pretrained(
  base_model_name, 
  trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

####################
### Load Dataset ###
####################
train_dataset_name = "mlabonne/guanaco-llama2-1k"
train_dataset = load_dataset(train_dataset_name, split="train")

#########################################
### Load LoRA Configurations for PEFT ###
#########################################
peft_config = LoraConfig(
    lora_alpha = 16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

##############################
### Set Training Arguments ###
##############################
training_arguments = TrainingArguments(
    output_dir="./tuning_results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)


##########################
### Set SFT Parameters ###
##########################
trainer = SFTTrainer(
    model=llama_3,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# trainer2 = SFTTrainer(
#     model=llama_3,
#     tokenizer=tokenizer,
#     train_dataset=train_dataset,
#     dataset_text_field="text",
#     max_seq_length=None,
#     dataset_num_proc=2,
#     packing=False,  # To speed up training for shorter sequences
#     args=TrainingArguments(
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=4,
#         warmup_steps=5,
#         max_steps=800,
#         do_eval=True,
#         learning_rate=3e-4,
#         log_level="debug",
#         bf16=True,
#         logging_steps=10,
#         optim="adamw_8bit",
#         weight_decay=0.01,
#         lr_scheduler_type="linear",
#         seed=3407,
#         output_dir="outputs",
#         report_to='wandb',
#         warmup_ratio=0.3,
#     ),
# )
#######################
### Fine-Tune Model ###
#######################
trainer.train()


##################
### Save Model ###
##################
new_model = "tuned-llama-3-8b"
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

#################
### Try Model ###
#################
prompt = "What is a large language model?"
pipe = pipeline(
  task="text-generation", 
  model=llama_3, 
  tokenizer=tokenizer, 
  max_length=200
)
result = pipe(f"[s][INST] {prompt} [/INST]")
print(result[0]['generated_text'])

PEFT_MODEL = "xavi00007/llama-3-8b"

llama_3.push_to_hub(
    PEFT_MODEL, use_auth_token=True
)