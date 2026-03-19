from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from datasets import Dataset, load_dataset, load_from_disk
from string import Template
import argparse
import torch.distributed as dist
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--quantization", type=str, default="bf16", choices=["bf16", "fp16", "8bit", "4bit"], help="Precision / quantization mode")
    parser.add_argument("--teacher_update_freq", type=int, default=100, help="Update teacher snapshot every N optimizer steps")
    parser.add_argument("--separate_teacher", action="store_true", help="Load a separate teacher model (more memory, potentially faster on multi-GPU)")
    parser.add_argument("--dataset_name", type=str, default="tooluse", help="Dataset name", choices=["tooluse", "science", "dummy"])
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    return parser.parse_args()


def load_model(model_name, quantization):
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    elif quantization == "fp16":
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    else:  # bf16
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)


def get_peft_config(quantization):
    """LoRA is required for quantized models (QLoRA). Optional for bf16/fp16."""
    if quantization in ("4bit", "8bit"):
        from peft import LoraConfig, TaskType
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
        )
    return None

def load_dummy_dataset(path="data/dummy_data.json") -> Dataset:
    with open(path) as f:
        examples = json.load(f)
    data = {
        "prompt": [
            [{"role": "user", "content": e["question"]}]
            for e in examples
        ],
        "teacher_prompt": [
            [
                {"role": "user", "content": e["question"]},
                {"role": "user", "content": (
                    f"This is an example for a response to the question:\n{e['answer']}\n\n"
                    f"Now answer with a response of your own, including the thinking process."
                )},
            ]
            for e in examples
        ],
    }
    return Dataset.from_dict(data), None


def load_tooluse_dataset(seed=42) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_dir = 'data/tooluse_data/train_data'
    train_dataset = load_from_disk(train_dir) 

    def format_example(example):

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=example['prompt'], output_text='\n'.join(example['golden_response']))}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


def load_science_dataset(seed=42) -> Dataset:
    """Load and prepare science dataset with formatted prompts."""
    path = 'data/science_data/train_data'
    print(f"Loading science dataset from {path}")
    dataset = load_from_disk(path)

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {'role': 'user', 'content': teacher_prompt.substitute(
                    orig_content=example['messages'][1]['content'],
                    output_text=example['output_text']
                )},
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


def preview_dataset(dataset, n=2):
    print("\n" + "="*60)
    print(f"DATASET PREVIEW ({len(dataset)} examples total, showing {n})")
    print("="*60)
    for i in range(min(n, len(dataset))):
        example = dataset[i]
        print(f"\n--- Example {i+1} ---")
        print("STUDENT INPUT (no example):")
        for msg in example["prompt"]:
            print(f"  [{msg['role'].upper()}] {msg['content']}")
        print("\nTEACHER INPUT (with example):")
        for msg in example["teacher_prompt"]:
            print(f"  [{msg['role'].upper()}] {msg['content']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model_name, args.quantization)
    teacher_model = load_model(args.model_name, args.quantization) if args.separate_teacher else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    peft_config = get_peft_config(args.quantization)

    if args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed)
    elif args.dataset_name == "science":
        dataset, _ = load_science_dataset(args.seed)
    elif args.dataset_name == "dummy":
        dataset, _ = load_dummy_dataset()
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    preview_dataset(dataset)

    config = DistilConfig(
        seed=args.seed,
        use_vllm=False,
        use_transformers_paged=True,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=(args.quantization == "bf16"),
        fp16=(args.quantization == "fp16"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        max_prompt_length=512,
        max_completion_length=512,
        num_train_epochs=args.num_train_epochs,
        num_iterations=1,
        num_generations=1,
        save_steps=100,
        max_grad_norm=1,
        report_to="wandb",
        output_dir=args.output_dir,
        log_completions=False,
        sync_ref_model=False,
        teacher_update_freq=args.teacher_update_freq,
        num_loss_tokens_to_skip=3,
    )
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
