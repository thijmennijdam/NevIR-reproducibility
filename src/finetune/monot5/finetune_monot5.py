import torch
from torch.utils.data import Dataset
import argparse
import os 
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

class MonoT5Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f'Query: {sample[0]} Document: {sample[1]} Relevant:'
        sample = {
          'text': text,
          'labels': sample[2],
        }
        return sample


def rename_checkpoints(checkpoint_dir):
    # List all items in the directory
    items = os.listdir(checkpoint_dir)
    
    # Filter to only include directories starting with "checkpoint-"
    checkpoint_dirs = []
    for item in items:
        full_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(full_path) and item.startswith("checkpoint-"):
            # Extract the number after the dash
            parts = item.split('-')
            if len(parts) == 2 and parts[1].isdigit():
                checkpoint_number = int(parts[1])
                checkpoint_dirs.append((checkpoint_number, full_path))
    
    # Sort by the numeric value
    checkpoint_dirs.sort(key=lambda x: x[0])
    
    # Rename directories to have continuous numbering
    for i, (_, old_path) in enumerate(checkpoint_dirs, start=1):
        new_name = f"checkpoint-{i}"
        new_path = os.path.join(checkpoint_dir, new_name)
        os.rename(old_path, new_path)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='castorini/monot5-base-msmarco-10k', type=str, required=False,
                        help="Base model to fine tune.")
    parser.add_argument("--triples_path", default='data/NevIR_data/train_triplets.tsv', type=str, required=False,
                        help="Triples.tsv path")
    parser.add_argument("--output_model_path", default='models/checkpoints/monot5/finetune_nevir', type=str, required=False,
                        help="Path for trained model and checkpoints.")
    parser.add_argument("--save_every_n_steps", default=0, type=int, required=False,
                        help="Save every N steps. (recommended 10000)")
    parser.add_argument("--logging_steps", default=100, type=int, required=False,
                        help="Logging steps parameter.")
    parser.add_argument("--per_device_train_batch_size", default=8, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int, required=False,
                        help="Gradient accumulation parameter.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, required=False,
                        help="Learning rate parameter.")
    parser.add_argument("--epochs", default=20, type=int, required=False,
                        help="Number of epochs to train")

    device = torch.device('cuda')
    args = parser.parse_args()

    torch.manual_seed(123)
    
    # 
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    train_samples = []
    with open(args.triples_path, 'r', encoding="utf-8") as fIn:
        for num, line in enumerate(fIn):
            if num > 6.4e5 * args.epochs:
                break
            query, positive, negative = line.split("\t")
            train_samples.append((query, positive, 'true'))
            train_samples.append((query, negative, 'false'))

    def smart_batching_collate_text_only(batch):
        texts = [example['text'] for example in batch]
        tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
        tokenized['labels'] = tokenizer([example['labels'] for example in batch], return_tensors='pt')['input_ids']

        for name in tokenized:
            tokenized[name] = tokenized[name].to(device)

        return tokenized

    
    dataset_train = MonoT5Dataset(train_samples)

    if args.save_every_n_steps:
        steps = args.save_every_n_steps
        strategy = 'steps'
    else:
        steps = 1
        strategy = 'epoch'

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_path,
        do_train=True,
        save_strategy=strategy,
        save_steps =steps, 
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=5e-5,
        num_train_epochs=args.epochs,
        warmup_steps=100,
        adafactor=True,
        seed=1,
        disable_tqdm=False,
        load_best_model_at_end=False,
        predict_with_generate=True,
        dataloader_pin_memory=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset_train,
        tokenizer=tokenizer,
        data_collator=smart_batching_collate_text_only,
    )

    trainer.train()

    trainer.save_model(args.output_model_path)
    trainer.save_state()

    # we rename the checkpoints so they are saved based on epoch number, instead of len(dataset)/batch_size
    rename_checkpoints(args.output_model_path)
    
    
if __name__ == "__main__":
    main()