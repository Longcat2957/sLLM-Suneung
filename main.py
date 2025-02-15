import os
import time
import ollama
import shutil
import logging
import argparse
import subprocess

from tqdm import tqdm
from utility.misc import load_json, load_txt, natural_key, format_text
from utility.model import ChatSession

MODEL_DICT = load_json("cfg/models.json")
AVAILABLE_MODELS = list(MODEL_DICT.keys())

parser = argparse.ArgumentParser(description="sLLM vs Korea Suneung")
parser.add_argument("--model", type=str, default="gemma2:2b", choices=AVAILABLE_MODELS, help="Model to use")
parser.add_argument("--task", type=str, default="kor", choices=["kor", "math"], help="Task to solve")
parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
parser.add_argument("--debug", action="store_true", help="Debug mode")

# options
parser.add_argument("--seed", type=int, default=42, help="for reproducibility")
parser.add_argument("--mirostat", type=int, default=0, choices=[1, 2, 3], help="Mirostat for sampling")
parser.add_argument("--mirostat-eta", type=float, default=0.1, help="Mirostat eta for sampling")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
parser.add_argument("--top-k", type=int, default=40, help="Top-k for sampling")
parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for sampling")
parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repeat penalty for sampling")

if __name__ == "__main__":
    args = parser.parse_args()
    # set logging level
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
        
    logger = logging.getLogger(__name__)
    
    # load system prompt
    _sys_prompt_basename = args.task + "-sys-prompt.txt"
    _sys_prompt_path = os.path.join("sys-prompt", _sys_prompt_basename)
    if not os.path.exists(_sys_prompt_path):
        raise FileNotFoundError(f"System prompt file not found: {_sys_prompt_path}")
    sys_prompt = load_txt(_sys_prompt_path)
    
    # load problems
    _problems_dir = os.path.join("problems", args.task)
    if not os.path.exists(_problems_dir):
        raise FileNotFoundError(f"Problems directory not found: {_problems_dir}")
    problem_file_paths = [os.path.join(_problems_dir, x) for x in sorted(os.listdir(_problems_dir), key=natural_key)]
    
    # check result directory (e.g. results/)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # check result subdirectory (e.g. results/{args.model})
    _result_subdir = os.path.join(args.output_dir, args.model)
    if not os.path.exists(_result_subdir):
        os.makedirs(_result_subdir)
    
    # check result subdirectory (e.g. results/{args.model}/{args.task})
    _result_subdir = os.path.join(_result_subdir, args.task)
    if os.path.exists(_result_subdir):
        # remove existing files
        shutil.rmtree(_result_subdir)
    os.makedirs(_result_subdir)
    
    # get model properties
    # (e.g. model_info = {'context_length": 8192, "model": "gemma2:2b-instruct-q4_K_S"})
    model_info = MODEL_DICT[args.model]
    
    # pull model (ollama)
    try:
        ollama.pull(model_info["model"])
    except Exception as e:
        logger.error(f"Failed to pull model {args.model}: {str(e)}")
        exit(1)
        
    # before create chat session, setting modelfile
    # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    options ={
        "num_ctx": model_info["context_length"],
        "mirostat": args.mirostat,
        "mirostat_eta": args.mirostat_eta,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repeat_penalty": args.repeat_penalty
    }
    logger.debug(f"Options: {options}")
        
    # create chat session
    chat_session = ChatSession(model_info["model"], sys_prompt, options)
    
    start_time = time.time()
    for problem_path in tqdm(problem_file_paths):
        problem_basename = os.path.basename(problem_path)
        problem = load_txt(problem_path)
        try:
            response_text = chat_session.send(problem)
            formatted_text = format_text(response_text)
            response_txt_path = os.path.join(_result_subdir, problem_basename)
            # save response text
            with open(response_txt_path, "w") as f:
                f.write(formatted_text)
        
        except Exception as e:
            logger.warning(f"Failed to solve problem {problem_basename}: {str(e)}")
            continue
        
        # 대화 기록 초기화(system prompt만 남김)
        chat_session.clear()

    elapsed_time = time.time() - start_time
    
    # subprocess를 통해 기존 ollama 인스턴스를 종료
    try:
        subprocess.run(["ollama", "stop", f"{model_info['model']}"])
    except Exception as e:
        logger.error(f"Failed to stop model {args.model}: {str(e)}")
        exit(1)