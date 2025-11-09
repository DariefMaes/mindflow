from dataclasses import dataclass, field
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    set_seed,
)
import os
import subprocess
import boto3
from botocore.exceptions import ClientError
from huggingface_hub import login
import torch
import json
import random

from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from torch_xla.core.xla_model import is_master_ordinal
from optimum.neuron.models.training import NeuronModelForCausalLM

random.seed(23)

def training_function(script_args, training_args):
    #dataset = load_dataset("knkarthick/dialogsum", split="train")
    dataset = load_dataset("emdil99/convo_summary", split="train")
    dataset = dataset.shuffle(seed=23)
    # def is_short_enough(example):
    #     return len(example["dialogue"].split()) < 200
    # filtered_dataset = dataset.filter(is_short_enough)

    # # Split dynamically (90% train, 10% eval)
    # split = filtered_dataset.train_test_split(test_size=0.1, seed=23)
    # train_dataset = split["train"]
    # eval_dataset = split["test"]



    # compute split sizes dynamically
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size

    # use train_test_split for convenience
    split = dataset.train_test_split(test_size=eval_size, seed=23)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # data_list = []
    # with open("/home/ubuntu/environment/FineTuning/HuggingFaceExample/01_finetuning/assets/conversations.json", 'r') as f:
    #     data = json.load(f)
    #     #print("data: ", data)
    #     data_list.append(data)

    # print("datalist: ", data_list)
    # #dataset = Dataset.from_list(data)
    # random.shuffle(data_list)
    



    #dataset = dataset.shuffle(seed=23)
    #train_dataset = dataset.select(range(10))
    #eval_dataset = dataset.select(range(11, 20))

    # train_size = int(0.9 * len(data_list))
    # train_dataset = data_list[:train_size]
    # eval_dataset = data_list[train_size:]


    # TOPIC_TO_CATEGORY = {
    #     "school": "School",
    #     "personal": "Personal",
    #     "work": "Work",
    #     "health": "Health",
    #     "career": "Career",
    # }

    # def create_conversation(sample):
    #     system_message = (
    #         "You are a task management assistant. "
    #         "Users conversationally describe their plans, worries, and activities.\n"
    #         "Your job is to extract concrete to-dos from their message.\n"
    #         "Do not invent tasks that are not reasonably implied.\n"
    #         "Do not invent exact calendar dates if not stated by the user; "
    #         "preserve relative phrases like 'tonight', 'this weekend', or 'by Monday' when present.\n"
    #         "Recognize implied or tentative tasks (e.g., 'I should', 'I might try to'); "
    #         "include them with Medium or Low urgency.\n"
    #         "If there are no action items then output {\"todos\":[]}.\n"
    #         "Return ONLY valid JSON in the format:\n"
    #         "{\"todos\":[{\"task\": str, \"category\":\"School|Personal|Work|Health|Career\", "
    #         "\"urgency\":\"Low|Medium|High\"}]}"
    #     )

    #     # Map dataset topic → category label
    #     topic = (sample.get("topic") or "").lower()
    #     category = TOPIC_TO_CATEGORY.get(topic, "Personal")

    #     # Build target JSON from provided todos list
    #     raw_todos = sample.get("todos") or []
    #     if len(raw_todos) == 0:
    #         todos_obj = {"todos": []}
    #     else:
    #         todos_obj = {
    #             "todos": [
    #                 {
    #                     "task": t,
    #                     "category": category,
    #                     # Simple, consistent rule: label as Medium by default
    #                     # (you can later customize based on wording / position)
    #                     "urgency": "Medium",
    #                 }
    #                 for t in raw_todos
    #             ]
    #         }

    #     return {
    #         "messages": [
    #             {"role": "system", "content": system_message},
    #             {"role": "user", "content": sample["dialogue"]},
    #             {
    #                 "role": "assistant",
    #                 "content": json.dumps(todos_obj, ensure_ascii=False),
    #             },
    #         ]
    #     }











    def create_conversation(sample):
        system_message = (
            "You are a task management assistant. Users conversationally discuss their activities with you.\n"
            "From their input, output a 1-2 sentence summary.\n"
            "If there are no action items then output empty tasks.\n"
        )

        return {
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": sample["dialogue"],          # from DialogSum
                },
                {
                    "role": "assistant",
                    "content": sample["summary"],           # from DialogSum
                },
            ]
        }






    # def create_conversation(sample):
    #     system_message = (
    #         "You are a task management assistant. Users conversationally discuss their activities with you.\n"
    #         "From their input, output a 1-2 sentence summary\n"
    #         "If there are no action items then output empty tasks.\n"
    #     )
    #     return {
    #         "messages": [
    #             {
    #                 "role": "system",
    #                 "content": system_message,
    #             },
    #             {"role": "user", "content": sample["conversation"]},
    #             {"role": "assistant", "content": sample["targets"]["summary"]}
    #         ]
    #     }

    train_dataset = train_dataset.map(
        create_conversation, remove_columns=train_dataset.features, batched=False
    )
    eval_dataset = eval_dataset.map(
        create_conversation, remove_columns=eval_dataset.features, batched=False
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_id)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.eos_token_id = 128001

    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = NeuronModelForCausalLM.from_pretrained(
        script_args.model_id,
        trn_config,
        torch_dtype=dtype,
        # Use FlashAttention2 for better performance and to be able to use larger sequence lengths.
        use_flash_attention_2=False, #Because we are training a sequence lower than 2K for the workshop
    )

    config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "gate_proj",
            "v_proj",
            "o_proj",
            "k_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = training_args.to_dict()

    sft_config = NeuronSFTConfig(
        max_seq_length=1024,
        packing=True,
        **args,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        },
    )

    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()
    del trainer


@dataclass
class ScriptArguments:
    model_id: str = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub."
        },
    )
    tokenizer_id: str = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={"help": "The tokenizer used to tokenize text for fine-tuning."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA r value to be used during fine-tuning."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha value to be used during fine-tuning."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout value to be used during fine-tuning."},
    )
    secret_name: str = field(
        default="huggingface/token",
        metadata={"help": "AWS Secrets Manager secret name containing Hugging Face token."},
    )
    secret_region: str = field(
        default="us-west-2",
        metadata={"help": "AWS region where the secret is stored."},
    )


def get_secret(secret_name, region_name):
    """
    Retrieve a secret from AWS Secrets Manager by searching for secrets with the given name prefix.  
    This is specific to the workshop environment.
    """
    try:
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=region_name)
        
        # List secrets and find one that starts with the secret_name
        paginator = client.get_paginator('list_secrets')
        for page in paginator.paginate():
            for secret in page['SecretList']:
                if secret['Name'].startswith(secret_name):
                    response = client.get_secret_value(SecretId=secret['ARN'])
                    if 'SecretString' in response:
                        return response['SecretString']
        return None
    except ClientError:
        print("Could not retrieve secret from AWS Secrets Manager")
        return None

if __name__ == "__main__":
    parser = HfArgumentParser([ScriptArguments, NeuronTrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    # Check for Hugging Face token in environment variable
    hf_token = os.environ.get("HF_TOKEN")
    
    # If no token in environment, try to get it from AWS Secrets Manager
    if not hf_token:
        print("No Hugging Face token found in environment, checking AWS Secrets Manager...")
        hf_token = get_secret(script_args.secret_name, script_args.secret_region)
    
    # Login to Hugging Face if a valid token is found
    if hf_token:
        print("Logging in to Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("No valid Hugging Face token found, continuing without authentication")
    
    set_seed(training_args.seed)
    training_function(script_args, training_args)

    # Consolidate LoRA adapter shards, merge LoRA adapters into base model, save merged model
    if is_master_ordinal():
        input_ckpt_dir = os.path.join(
            training_args.output_dir, f"checkpoint-{training_args.max_steps}"
        )
        output_ckpt_dir = os.path.join(training_args.output_dir, "merged_model")
        # the spawned process expects to see 2 NeuronCores for consolidating checkpoints with a tp=2
        # Either the second core isn't really used or it is freed up by the other thread finishing.  
        # Adjusting Neuron env. var to advertise 2 NeuronCores to the process.
        env = os.environ.copy()
        env["NEURON_RT_VISIBLE_CORES"] = "0-1"
        subprocess.run(
            [
                "python3",
                "consolidate_adapter_shards_and_merge_model.py",
                "-i",
                input_ckpt_dir,
                "-o",
                output_ckpt_dir,
            ],
            env=env
        )