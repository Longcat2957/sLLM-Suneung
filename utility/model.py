import ollama

class ChatSession:
    """
    ollama 대화 API를 래핑하는 클래스입니다.
    시스템 프롬프트를 초기 메시지로 추가하고, 대화 기록을 관리합니다.
    """
    def __init__(self, model: str, system_prompt: str, options: dict):
        self.model = model
        # 시스템 프롬프트 유지
        self.messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # 대화 옵션 설정
        self.options = options
        
    def send(self, user_message: str) -> str:
        self.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )
        response = ollama.chat(self.model, messages=self.messages, options=self.options)
        assistant_text = response.content if hasattr(response, "content") else str(response)
        self.messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text
    
    def clear(self):
        # 시스템 프롬프트만 남김
        self.messages = [self.messages[0]]