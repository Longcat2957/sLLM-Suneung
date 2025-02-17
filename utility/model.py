from vllm import LLM, SamplingParams, RequestOutput

def get_gguf_inference(model_path, tokenizer):
    # create an LLM via vLLM
    try:
        llm = LLM(model_path, tokenizer=tokenizer)
        return llm
    except Exception as e:
        raise ValueError(f"Failed to load model(vllm): {str(e)}")
    
class ChatSession:
    def __init__(self, llm: LLM, sampling_params: SamplingParams, system_prompt: str):
        self.llm = llm
        self.sampling_params = sampling_params
        self.system_prompt = {
            "role": "system",
            "content": system_prompt
        }
        
    def send(self, user_input: str):
        # send user input
        user_input = {
            "role": "user",
            "content": user_input
        }
        # returns A list of ``RequestOutput`` objects
        responses: list[RequestOutput] = self.llm.chat([self.system_prompt, user_input], self.sampling_params)
        response = responses[0].outputs[0].text
        return response