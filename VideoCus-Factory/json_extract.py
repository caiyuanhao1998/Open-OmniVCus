import json
import argparse
from tqdm import tqdm


def extract_keys(input_json, output_json, keys=("original", "vid")):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "Input JSON must be a list of dicts"

    new_data = []
    for item in tqdm(data, desc="Processing"):
        if not isinstance(item, dict):
            continue
        new_item = {k: item[k] for k in keys if k in item}
        new_data.append(new_item)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input json path")
    parser.add_argument("--output", required=True, help="output json path")
    args = parser.parse_args()

    extract_keys(args.input, args.output)
