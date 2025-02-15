import argparse
from utility.misc import load_json, load_txt

MODEL_DICT = load_json("cfg/models.json")
AVAILABLE_MODELS = list(MODEL_DICT.keys())

parser = argparse.ArgumentParser(description="sLLM vs Korea Suneung")
parser.add_argument("--model", type=str, default="gemma2:2b", choices=AVAILABLE_MODELS, help="Model to use")
parser.add_argument("--task", type=str, default="kor", choices=["kor", "math"], help="Task to solve")

if __name__ == "__main__":
    args = parser.parse_args()
    print(MODEL_DICT)