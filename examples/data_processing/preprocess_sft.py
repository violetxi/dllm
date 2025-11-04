""" Take any huggingface dataset and preprocess it for SFT format.
"""
import argparse
from datasets import Dataset, DatasetDict, load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--src_path", type=str, required=True)
parser.add_argument("--target_path", type=str, required=True)
args = parser.parse_args()

def alpaca_map_fn(sample):
    instruction = sample.pop("instruction")
    input = sample.pop("input")
    output = sample.pop("output")
    if input:
        prompt = f"{instruction}\n{input}"
    else:
        prompt = instruction
    messages = [{"role": "user","content": prompt}, 
                {"role": "assistant","content": output}]
    sample["messages"] = messages
    sample["source"] = "alpaca-cleaned"
    return sample

def preprocess_sft_dataset(src_path: str, target_path: str):
    ds = load_dataset(src_path)
    # split into train and test
    if "test" not in ds:
        ds = ds["train"]
        ds = ds.train_test_split(test_size=0.05, seed=42)
        ds = DatasetDict({
            "train": ds["train"],
            "test": ds["test"],
        })
    ds = ds.map(alpaca_map_fn)
    ds.push_to_hub(target_path, private=True)
    print(f"Saved to: {target_path}")
    print(ds)


if __name__ == "__main__":
    preprocess_sft_dataset(args.src_path, args.target_path)
