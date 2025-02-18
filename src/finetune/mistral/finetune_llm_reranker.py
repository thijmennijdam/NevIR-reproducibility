# Import necessary libraries
import os
import argparse
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

def load_and_format_dataset(dataset_name, tokenizer, split="train"):
    """
    Load and format the dataset with the desired prompt structure.
    """
    dataset = load_dataset(dataset_name, split=split)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False, 
            )
            for convo in convos
        ]
        return {"text": texts}

    return dataset.map(formatting_prompts_func, batched=True)

def initialize_model_and_tokenizer(model_name, max_seq_length, dtype, load_in_4bit):
    """
    Load the model and tokenizer with appropriate configurations.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Apply LoRA configurations
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Configure tokenizer for chat templates
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token=True,
    )

    return model, tokenizer

def train_model(model, tokenizer, train_dataset, save_path, batch_size, gradient_accumulation_steps, epochs, dataset_length):
    """
    Train the model using the SFTTrainer.
    """
    # Calculate total training steps based on batch size, gradient accumulation, and dataset length
    steps_per_epoch = (dataset_length // (batch_size * gradient_accumulation_steps)) + 1
    max_steps = steps_per_epoch * epochs

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=save_path,
            report_to="none",
            save_strategy="epoch",  # Save model at the end of each epoch
        ),
    )
    trainer.train()
    return trainer

def parse_arguments():
    """
    Parse command-line arguments for the training script.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a language model")
    parser.add_argument("--dataset_name", type=str, default="thijmennijdam/NevIR-conversational", 
                        help="Name of the dataset to use")
    parser.add_argument("--output_dir", type=str, default="/scratch-shared/tnijdam_ir2/finetuned_excluir", 
                        help="Base output directory for saving model")
    parser.add_argument("--model_name", type=str, default="unsloth/mistral-7b-instruct-v0.3", 
                        help="Name of the model to fine-tune")
    parser.add_argument("--max_seq_length", type=int, default=2048, 
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Number of gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--dtype", default=None, 
                        help="Data type (None for auto detection)")
    parser.add_argument("--load_in_4bit", action="store_true", 
                        help="Use 4-bit quantization")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()

    # Construct save path based on dataset name
    save_path = os.path.join(args.output_dir, args.dataset_name.split('/')[-1])
    os.makedirs(save_path, exist_ok=True)

    # Step 1: Initialize model and tokenizer
    print("Initializing model and tokenizer...")
    model, tokenizer = initialize_model_and_tokenizer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit
    )

    # Step 2: Load dataset
    print("Loading and formatting dataset...")
    train_dataset = load_and_format_dataset(args.dataset_name, tokenizer, split="train")
    
    # Step 3: Train the model
    print("Training the model...")
    trainer = train_model(
        model, 
        tokenizer, 
        train_dataset, 
        save_path,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        dataset_length=len(train_dataset)
    )
    
    # Save final model
    trainer.save_model(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()