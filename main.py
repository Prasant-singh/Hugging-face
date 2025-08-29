from transformers import (
    AutoTokenizer, GPT2LMHeadModel, AutoConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset, DatasetDict
import math

# Constants
context_length = 128
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Required for padding

# Load datasets
ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

raw_datasets = DatasetDict({
    "train": ds_train,
    "valid": ds_valid,
})

# Tokenization function
def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}



if __name__ == "__main__":
    # Tokenize and cache
    tokenized_datasets = raw_datasets.map(
        tokenize,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=4,  # Parallel preprocessing
        load_from_cache_file=True
    )

    # Model config and initialization
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training arguments
    args = TrainingArguments(
        output_dir="codeparrot-ds",
        per_device_train_batch_size=32,  # Large batch size for GPU
        gradient_accumulation_steps=4,   # Effective batch size = 256
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5000,
        logging_steps=1000,
        fp16=False,  # Mixed precision for speed
        report_to="none",  # Disable logging integrations
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer  # Still supported in v4.x
    )

    # Train
    trainer.train()

    # Save model
    output_dir = "./Model_final"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

    # Evaluate
    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"Perplexity: {perplexity:.2f}")
