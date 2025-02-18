import argparse
import os
import json
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
import re

def load_model(checkpoint_path, max_seq_length=512, dtype=None, load_in_4bit=False): #TODO: test with bigger seq length and False
    """Load the fine-tuned model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token=True,
    )
    return model, tokenizer

def extract_final_ranking(decoded_output):
    """
    Extracts the final ranking from the model's output, which follows the format:
    '[1] > [2]' or '[2] > [1]'.
    
    Args:
        decoded_output (str): The raw generated output from the model.
        
    Returns:
        str: The extracted ranking in the format '[X] > [Y]' or '[Y] > [X]'.
    """
    # Strip any leading or trailing spaces to avoid issues with the search
    decoded_output = decoded_output.strip()

    # Define a regex pattern to match the final ranking format like '[1] > [2]'
    # The final ranking should be after the last [INST] and the [2] > [1] type pattern.
    # We want to exclude any example ranking that may be part of the prompt
    ranking_pattern = r"\[\d+\] > \[\d+\](?=\s*</s>$)"

    # Search for the ranking pattern in the decoded output
    match = re.search(ranking_pattern, decoded_output)

    if match:
        return match.group(0)  # Return the matched ranking
    else:
        return "Ranking not found"

def perform_inference(model, tokenizer, dataset):
    """Perform inference on the dataset."""
    outputs = []
    for idx, example in tqdm(enumerate(dataset), desc="Performing inference"):
        messages = example["conversations"][:-1]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        
        generated = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True, do_sample=False, temperature=0.0)
        decoded_output = tokenizer.batch_decode(generated, skip_special_tokens=False)
        # print(decoded_output)
        assistant_output = extract_final_ranking(decoded_output[0])
        print(assistant_output)
        outputs.append({
            "index": idx,
            "input": messages,
            "raw_output": decoded_output,
            "assistant_output": assistant_output,
        })
        
    return outputs

def evaluate_outputs_nevir(outputs):
    """
    Evaluate the model's outputs by pairing consecutive samples.
    For (even_index, odd_index):
    - Even index should output "[1] > [2]"
    - Odd index should output "[2] > [1]"

    Returns the average score and pairwise accuracy.
    """
    scores = []
    num_pairs = len(outputs) // 2  # Number of pairs to evaluate

    for i in range(num_pairs):
        even_idx = i * 2
        odd_idx = even_idx + 1

        # Extract assistant outputs
        even_output = outputs[even_idx]["assistant_output"]
        odd_output = outputs[odd_idx]["assistant_output"]

        # Check the expected responses
        even_expected = "[1] > [2]"
        odd_expected = "[2] > [1]"

        # Scoring for the even index
        even_score = 1.0 if even_expected in even_output else 0.0

        # Scoring for the odd index
        odd_score = 1.0 if odd_expected in odd_output else 0.0

        # Append scores for this pair
        if even_score and odd_score:
            scores.append(1)
        else:
            scores.append(0)
        
    # Calculate average and pairwise accuracy
    pairwise_accuracy = sum(scores) / len(scores)

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

    return correct / len(outputs)

def evaluate_outputs(outputs, dataset_name):
    if "NevIR" in dataset_name:
        return evaluate_outputs_nevir(outputs)
    elif "ExcluIR" in dataset_name:
        return evaluate_outputs_excluir(outputs)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def save_results(output_path, outputs):
    """Save outputs and evaluation results to the output directory."""
    # Create a subdirectory for the checkpoint name

    # Save outputs
    outputs_file = os.path.join(output_path, "outputs.json")
    with open(outputs_file, "w") as f:
        json.dump(outputs, f, indent=4)

    print(f"Outputs saved to: {outputs_file}")
    return outputs_file


def load_saved_outputs(output_path):
    """Load previously saved outputs from the output directory."""
    # Define the file path for saved outputs
    outputs_file = os.path.join(output_path, "outputs.json")

    # Check if the file exists
    if not os.path.exists(outputs_file):
        raise FileNotFoundError(f"Saved outputs not found at {outputs_file}. Run without --skip_inference first.")

    # Load and return the outputs
    with open(outputs_file, "r") as f:
        outputs = json.load(f)
    return outputs

def sorted_checkpoints(checkpoint_dir):
    """Sorts checkpoint directories based on the step number."""
    checkpoint_names = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-")]

    def extract_step(name):
        match = re.search(r"checkpoint-(\d+)", name)
        if match:
            return int(match.group(1))
        return -1  # Handle cases where the name doesn't match

    return sorted(checkpoint_names, key=extract_step)

# Main Script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, help="Path to a single checkpoint.")
    parser.add_argument("--checkpoint_dir_base", type=str, help="Path to the directory containing multiple checkpoints.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to evaluate on.")
    parser.add_argument("--output_dir_base", type=str, default="results/mistral", help="Directory to save outputs and results.")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference and load saved outputs.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on.")

    args = parser.parse_args()
    
    if not args.checkpoint_path and not args.checkpoint_dir_base:
        raise ValueError("You must provide either --checkpoint_path or --checkpoint_dir_base.")

    if args.checkpoint_path and args.checkpoint_dir_base:
        raise ValueError("Specify only one: --checkpoint_path or --checkpoint_dir_base.")


    if "nevir" in args.dataset_name.lower():
        args.output_dir_base += "/evaluate_nevir"
        dataset_name = "thijmennijdam/NevIR-conversational"
    elif "excluir" in args.dataset_name.lower():
        args.output_dir_base += "/evaluate_excluir"
        dataset_name = "thijmennijdam/ExcluIR-conversational"
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
    
    print(f"Output directory: {args.output_dir_base}")
    os.makedirs(args.output_dir_base, exist_ok=True)
    csv_path = os.path.join(args.output_dir_base, "evaluation_results.csv")
    
    with open(csv_path, "w") as csv_file:
        csv_file.write("checkpoint,accuracy\n")
    
    checkpoint_list = (
        [args.checkpoint_path] if args.checkpoint_path 
        else [os.path.join(args.checkpoint_dir_base, ckpt) for ckpt in sorted_checkpoints(args.checkpoint_dir_base)]
    )
    
    for idx, checkpoint_path in enumerate(checkpoint_list):
        output_path = os.path.join(args.output_dir_base, os.path.basename(checkpoint_path))
        os.makedirs(output_path, exist_ok=True)
        
        print(f"Evaluating checkpoint: {checkpoint_path}")
        
        if args.skip_inference:
            # Load saved outputs
            print("Loading saved outputs...")
            outputs = load_saved_outputs(output_path)
        else:
            # Step 1: Load model
            print("Loading model...")
            model, tokenizer = load_model(checkpoint_path)

            # Step 2: Load dataset
            print("Loading dataset...")
            eval_dataset = load_dataset(dataset_name, split=args.split, download_mode='force_redownload')

            # Step 3: Perform inference
            print("Performing inference...")
            outputs = perform_inference(model, tokenizer, eval_dataset)

        # Step 4: Save outputs of the model
        print(f"Saving outputs to: {output_path}")
        save_results(output_path, outputs)

        # Step 5: Evaluate outputs, based on the dataset we evaluate slighty differently
        print("Evaluating outputs...")
        accuracy = evaluate_outputs(outputs, dataset_name)

        # Step 6: Save evaluation results
        with open(csv_path, "a") as csv_file:
            csv_file.write(f"{idx},{accuracy}\n")

        print(f"Accuracy: {accuracy:.2f}")