import os
import logging
import argparse

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from utility.misc import load_yaml, load_txt
from utility.model import get_gguf_inference, ChatSession

MODEL_DICT = load_yaml("cfg/models.yaml")
AVAILABLE_MODELS = list(MODEL_DICT.keys())

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gemma2:2b", choices=AVAILABLE_MODELS, help="Model to use")
parser.add_argument("--quant_type", type=str, default="Q4_K_M", help="Quantization type")
parser.add_argument("--task", type=str, default="kor", choices=["kor", "math"], help="Task to solve")
parser.add_argument("--output_dir", type=str, default="results", help="Output directory")

# Sampling Parameters
parser.add_argument("--n", type=int, default=1, help="Number of output sequences to return for the given prompt.")
parser.add_argument("--best_of", type=int, default=1, help="Number of output sequences generated from the prompt (must be >= n).")
parser.add_argument("--presence_penalty", type=float, default=0.0, help="Penalizes new tokens based on whether they already appear in the generated text.")
parser.add_argument("--frequency_penalty", type=float, default=0.0, help="Penalizes new tokens based on their frequency in the generated text so far.")
parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Penalizes tokens that appear in the prompt and generated text. Values > 1 favor new tokens.")
parser.add_argument("--temperature", type=float, default=1.0, help="Controls randomness in sampling. 0 means greedy sampling.")
parser.add_argument("--top_p", type=float, default=1.0, help="Cumulative probability threshold for token selection (0 < top_p <= 1).")
parser.add_argument("--top_k", type=int, default=-1, help="Number of top tokens to consider. Use -1 to consider all tokens.")
parser.add_argument("--min_p", type=float, default=0.0, help="Minimum probability threshold for a token relative to the highest probability token.")
parser.add_argument("--seed", type=int, default=None, help="Random seed for generation.")
parser.add_argument("--stop", nargs="+", default=[], help="List of stop strings. Generation stops when one of these is produced.")
parser.add_argument("--stop_token_ids", nargs="+", type=int, default=[], help="List of token ids that cause generation to stop.")
parser.add_argument("--bad_words", nargs="+", default=[], help="List of words that are forbidden in the generated output.")
parser.add_argument("--include_stop_str_in_output", action="store_true", help="Include stop strings in the generated output text.")
parser.add_argument("--ignore_eos", action="store_true", help="Continue generating tokens even after the EOS token is produced.")
parser.add_argument("--max_tokens", type=int, default=-1, help="Maximum number of tokens to generate per output sequence.")
parser.add_argument("--min_tokens", type=int, default=0, help="Minimum number of tokens to generate before allowing stop conditions.")
parser.add_argument("--logprobs", type=int, default=None, help="Number of log probabilities to return per output token. If set, the result includes the log probabilities of the top tokens.")
parser.add_argument("--prompt_logprobs", type=int, default=None, help="Number of log probabilities to return per prompt token.")
parser.add_argument("--detokenize", action="store_true", help="Detokenize the output text (default behavior).")
parser.add_argument("--skip_special_tokens", action="store_true", help="Skip special tokens in the output text.")
parser.add_argument("--spaces_between_special_tokens", action="store_true", help="Add spaces between special tokens in the output (default behavior).")
parser.add_argument("--truncate_prompt_tokens", type=int, default=None, help="If set, use only the last k tokens from the prompt.")
parser.add_argument("--guided_decoding", type=str, default=None, help="Guided decoding parameters as a JSON string.")
parser.add_argument("--logit_bias", type=str, default=None, help="Logit bias settings as a JSON string.")
parser.add_argument("--allowed_token_ids", nargs="+", type=int, default=None, help="List of allowed token ids; only these tokens are retained in logits.")

if __name__ == "__main__":
    args = parser.parse_args()
    
    model_info = MODEL_DICT[args.model]
    
    # (1) 사용자가 지정한 quant_type이 해당 모델에서 지원되는지 확인합니다.
    if args.quant_type not in model_info["available_quant_type"]:
        raise ValueError(f"{args.model} 모델은 {args.quant_type} 양자화를 지원하지 않습니다.")
    
    # (2) 모델에 맞는 tokenizer 로드
    tokenizer = model_info['tokenizer']
    
    # (3) huggingface URL과 quant_type을 이용해 모델 파일 경로를 구성
    repo_id = model_info["repo_id"]
    filename = model_info["available_quant_type"][args.quant_type]
    model = hf_hub_download(repo_id, filename=filename)
    
    # (4) SamplingParams 객체 생성
    sampling_params = SamplingParams(
        n=args.n,
        best_of=args.best_of,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.seed,
        stop=args.stop,
        stop_token_ids=args.stop_token_ids,
        bad_words=args.bad_words,
        include_stop_str_in_output=args.include_stop_str_in_output,
        ignore_eos=args.ignore_eos,
        max_tokens=model_info["context_length"] if args.max_tokens < 0 else args.max_tokens,
        min_tokens=args.min_tokens,
        logprobs=args.logprobs,
        prompt_logprobs=args.prompt_logprobs,
        detokenize=args.detokenize,
        skip_special_tokens=args.skip_special_tokens,
        spaces_between_special_tokens=args.spaces_between_special_tokens,
        truncate_prompt_tokens=args.truncate_prompt_tokens,
        guided_decoding=args.guided_decoding,
        logit_bias=args.logit_bias,
        allowed_token_ids=args.allowed_token_ids
    )
    
    # (5) 시스템 프롬프트
    system_prompt_basename = args.task + "-sys-prompt.txt"
    system_prompt_path = os.path.join("sys-prompt", system_prompt_basename)
    if not os.path.exists(system_prompt_path):
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")
    system_prompt = load_txt(system_prompt_path)
    
    # (6) LLM(vllm) 객체 생성
    llm = get_gguf_inference(model, tokenizer)
    
    # (7) ChatSession 객체 생성
    chat_session = ChatSession(llm, sampling_params, system_prompt)
    
    # TESTTESTTEST
    
    sample_kor_problem = """
    
    <지문>
밑줄 긋기는 일상적으로 유용하게 활용할 수 있는 독서 전략
이다. 밑줄 긋기는 정보를 머릿속에 저장하고 기억한 내용을 
떠올리는 데 도움이 된다. 독자로 하여금 표시한 부분에 주의를
기울이도록 해 정보를 머릿속에 저장하도록 돕고, 표시한 부분이
독자에게 시각적 자극을 주어 기억한 내용을 떠올리는 데 단서가
되기 때문이다. 이러한 점에서 밑줄 긋기는 일반적인 독서 
상황뿐 아니라 학습 상황에서도 유용하다. 또한 밑줄 긋기는 
방대한 정보들 가운데 주요한 정보를 추리는 데에도 효과적이며, 
표시한 부분이 일종의 색인과 같은 역할을 하여 독자가 내용을 
다시 찾아보는 데에도 용이하다.

밑줄 긋기의 효과를 얻기 위한 방법에는 몇 가지가 있다. 우선
글을 읽는 중에는 문장이나 문단에 나타난 정보 간의 상대적 
중요도를 결정할 때까지 밑줄 긋기를 잠시 늦추었다가 주요한 
정보에 밑줄 긋기를 한다. 이때 주요한 정보는 독서 목적에 따라
달라질 수 있다는 점을 고려한다. 또한 자신만의 밑줄 긋기 표시
체계를 세워 밑줄 이외에 다른 기호도 사용할 수 있다. 밑줄 
긋기 표시 체계는 밑줄 긋기가 필요한 부분에 특정 기호를 
사용하여 표시하기로 독자가 미리 정해 놓는 것이다. 예를 들면
하나의 기준으로 묶을 수 있는 정보들에 동일한 기호를 붙이거나
순차적인 번호를 붙이기로 하는 것 등이다. 이는 기본적인 밑줄
긋기를 확장한 방식이라 할 수 있다. 
밑줄 긋기는 어떠한 수준의 독자라도 쉽게 사용할 수 있다는 
점 때문에 연습 없이 능숙하게 사용할 수 있다고 오해되어 온 
경향이 있다. 그러나 본질적으로 밑줄 긋기는 주요한 정보가 
무엇인지에 대한 판단이 선행되어야 한다는 점에서 단순하지 
않다. ㉠<u>밑줄 긋기의 방법을 이해하고 잘 사용하는 것</u>은 글을 
능동적으로 읽어 나가는 데 도움이 될 수 있다
</지문>

<문제>
윗글의 내용과 일치하지 않는 것은?
</문제>

<객관식답안>
① 밑줄 긋기는 일반적인 독서 상황에서 도움이 된다.
② 밑줄 이외의 다른 기호를 밑줄 긋기에 사용하는 것이 가능하다.
③ 밑줄 긋기는 누구나 연습 없이도 능숙하게 사용할 수 있는 
전략이다.
④ 밑줄 긋기로 표시한 부분은 독자가 내용을 다시 찾아보는 데 
유용하다.
⑤ 밑줄 긋기로 표시한 부분이 독자에게 시각적인 자극을 주어 
기억한 내용을 떠올리는 데 도움이 된다.
</객관식답안>
    
    
    """
    print(chat_session.send(sample_kor_problem))