import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load all variables from .env automatically
load_dotenv()


class Generator:
    """
    Unified LLM Generator for multiple OpenAI-compatible providers:
    - OpenAI
    - Anthropic 
    - Google Gemini
    - Groq
    - DeepSeek

    
    Example:
        gen = Generator(provider="openai")
        response = gen.generate(
            system_prompt="You are a helpful assistant that explains concepts clearly.",
            user_prompt="Explain retrieval-augmented generation (RAG) in simple terms."
        )
        print(response)
    """

    def __init__(
        self,
        provider: str,
        model_name: str = None,
        max_tokens: int = None,
        temperature: float = 0,
        timeout: int = None,
        max_retries: int = 2,
        top_p: float = 1.0,  
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.top_p = top_p   

        # Initialize the correct client
        self.client = self._init_client()
        

    def _init_client(self):
        if self.provider == "openai":
            return ChatOpenAI(
                model_name=self.model_name or "gpt-4o-mini",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                max_retries=self.max_retries,
                top_p=self.top_p,   
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        elif self.provider == "gemini":
            return ChatOpenAI(
                model_name=self.model_name or "gemini-2.5-flash",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                max_retries=self.max_retries,
                top_p=self.top_p,
                api_key=os.getenv("GOOGLE_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )

        elif self.provider == "groq":
            return ChatOpenAI(
                model_name=self.model_name or "llama-3.1-8b-instant",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                max_retries=self.max_retries,
                top_p=self.top_p,
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )

        elif self.provider == "anthropic":
            return ChatOpenAI(
                model_name=self.model_name or "claude-3-opus-20240229",
                temperature=self.temperature,
                max_tokens=self.max_tokens or 120,
                timeout=self.timeout,
                max_retries=self.max_retries,
                top_p=self.top_p,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                base_url="https://api.anthropic.com/v1/",
            )

        elif self.provider == "deepseek":
            return ChatOpenAI(
                model_name=self.model_name or "deepseek-chat",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                max_retries=self.max_retries,
                top_p=self.top_p,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
            )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a system + user prompt to the LLM and return the response.

        Args:
            system_prompt (str): The system instruction (e.g. behavior, role).
            user_prompt (str): The actual user query.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.client.invoke(messages)
        return response.content.strip()

