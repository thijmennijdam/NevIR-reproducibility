import argparse
import os
import json
from tqdm import tqdm
import re
import torch
from datasets import load_dataset
from transformers import pipeline

def load_model(checkpoint_path):
    """Load the fine-tuned model as a text-generation pipeline."""
    print("CHECKPOINT PATH:", checkpoint_path)
    generator = pipeline(
        "text-generation",
        model=checkpoint_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map='auto'
    )
    return generator

def perform_inference(dataset, generator=None):
    """Perform inference on the dataset."""
    outputs = []
    i = 0

    for idx, example in tqdm(enumerate(dataset), desc="Performing inference"):
        messages = example["conversations"][:-1]
        
        # Print for debugging
        # print("ORIGINAL MESSAGES:", messages)

        # We'll map from your dataset's structure to a conversation structure
        mapping = {
            "role": "from",
            "content": "value",
            "user": "human",
            "assistant": "gpt"
        }
        mapped_messages = []
        for msg in messages:
            # Convert "human" -> "user" or "assistant" -> "assistant" as needed
            original_role = msg[mapping["role"]]
            if original_role == mapping["user"]:      # "human"
                new_role = "user"
            elif original_role == mapping["assistant"]:  # "gpt"
                new_role = "assistant"
            else:
                new_role = original_role  # fallback

            mapped_messages.append({
                "role": new_role,
                "content": msg[mapping["content"]]
            })

        # print("MAPPED MESSAGES:", mapped_messages)
        
        # If your pipeline supports a list-of-dicts "messages" format:
        generation = generator(
            mapped_messages,
            do_sample=False,
            top_p=1,
            max_new_tokens=10
        )
        
        try:
            response = generation[0]["generated_text"][-1]["content"]
        except (KeyError, IndexError, TypeError):
            response = generation[0].get("generated_text", "")

        # print("RESPONSE:", response)
        
        outputs.append({
            "index": idx,
            "input": messages,
            "raw_output": response,
            "assistant_output": response,
        })

    return outputs

def evaluate_outputs_nevir(outputs):
    """
    Evaluate the model's outputs by pairing consecutive samples.
    For (even_index, odd_index):
    - Even index should output "[1] > [2]"
    - Odd index should output "[2] > [1]"
    """
    scores = []
    num_pairs = len(outputs) // 2

    for i in range(num_pairs):
        even_idx = i * 2
        odd_idx = even_idx + 1

        even_output = outputs[even_idx]["assistant_output"]
        odd_output = outputs[odd_idx]["assistant_output"]

        # Check the expected responses
        even_expected = "[1] > [2]"
        odd_expected = "[2] > [1]"

        even_score = 1.0 if even_expected in even_output else 0.0
        odd_score = 1.0 if odd_expected in odd_output else 0.0

        if even_score and odd_score:
            scores.append(1)
        else:
            scores.append(0)
        
    pairwise_accuracy = sum(scores) / len(scores) if scores else 0.0
    return pairwise_accuracy

def evaluate_outputs_excluir(outputs):
    """Evaluate the model's outputs and compute accuracy."""
    correct = 0
    even_expected = "[1] > [2]"
    odd_expected = "[2] > [1]"

    for i, output in enumerate(outputs):
        if i % 2 == 0:
            if even_expected in output["assistant_output"]:
                correct += 1
        else:
            if odd_expected in output["assistant_output"]:
                correct += 1

    return correct / len(outputs) if outputs else 0.0

def evaluate_outputs(outputs, dataset_name):
    """Dispatch to the correct evaluation function."""
    if "NevIR" in dataset_name:
        return evaluate_outputs_nevir(outputs)
    elif "ExcluIR" in dataset_name:
        return evaluate_outputs_excluir(outputs)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def save_results(output_path, outputs):
    """Save outputs and evaluation results to the output directory."""
    os.makedirs(output_path, exist_ok=True)
    outputs_file = os.path.join(output_path, "outputs.json")
    with open(outputs_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4)
    print(f"Outputs saved to: {outputs_file}")
    return outputs_file

def load_saved_outputs(output_path):
    """Load previously saved outputs from the output directory."""
    outputs_file = os.path.join(output_path, "outputs.json")
    if not os.path.exists(outputs_file):
        raise FileNotFoundError(
            f"Saved outputs not found at {outputs_file}. "
            "Run without --skip_inference first."
        )
    with open(outputs_file, "r", encoding="utf-8") as f:
        outputs = json.load(f)
    return outputs

def sorted_checkpoints(checkpoint_dir):
    """Sorts checkpoint directories based on the step number."""
    checkpoint_names = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-")
    ]
    def extract_step(name):
        match = re.search(r"checkpoint-(\d+)", name)
        if match:
            return int(match.group(1))
        return -1  # Handle cases where the name doesn't match
    return sorted(checkpoint_names, key=extract_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir_base", type=str, required=True,
                        help="Path to the checkpoint to be loaded.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset to evaluate on.")
    parser.add_argument("--output_dir_base", type=str, default="results/mistral",
                        help="Directory to save outputs and results.")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference and load saved outputs.")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate on.")
    
    args = parser.parse_args()

    # Decide which HF dataset to load based on argument
    if "nevir" in args.dataset_name.lower():
        args.output_dir_base += "/evaluate_nevir"
        dataset_name = "thijmennijdam/NevIR-conversational"
    elif "excluir" in args.dataset_name.lower():
        args.output_dir_base += "/evaluate_excluir"
        dataset_name = "thijmennijdam/ExcluIR-conversational"
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    # Create the output directory
    output_path = args.output_dir_base
    os.makedirs(output_path, exist_ok=True)

    # Prepare a CSV file to log results
    csv_path = os.path.join(output_path, "evaluation_results.csv")
    with open(csv_path, "w", encoding="utf-8") as csv_file:
        csv_file.write("checkpoint,accuracy\n")

    if args.skip_inference:
        # Load saved outputs
        print("Loading saved outputs...")
        outputs = load_saved_outputs(output_path)
    else:
        # Load the pipeline (generator)
        print("Loading model...")
        generator = load_model(args.checkpoint_dir_base)

        # Load dataset
        print("Loading dataset...")
        eval_dataset = load_dataset(dataset_name, split=args.split, download_mode='force_redownload')

        # Perform inference
        print("Performing inference...")
        outputs = perform_inference(eval_dataset, generator=generator)

        # Save results
        save_results(output_path, outputs)

    # Evaluate outputs
    print("Evaluating outputs...")
    accuracy = evaluate_outputs(outputs, dataset_name)

    # Save evaluation results to CSV
    with open(csv_path, "a", encoding="utf-8") as csv_file:
        # If you want to track multiple checkpoints, 
        # you could loop over them and write each result. 
        # Here we use the single `checkpoint_dir_base` as ID:
        csv_file.write(f"{os.path.basename(args.checkpoint_dir_base)},{accuracy:.4f}\n")

    print(f"Accuracy: {accuracy:.4f}")
    print("Done.")
