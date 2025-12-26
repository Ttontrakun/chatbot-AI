# llm_factory.py
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

def get_llm(name: str):
    if name == "typhoon":
        return ChatOpenAI(
            base_url="https://api.typhoon.ai",
            api_key="YOUR_KEY",
            model="typhoon-v1.5",
            temperature=0
        )

    if name == "qwen":
        return ChatOllama(
            model="qwen2.5:7b",
            temperature=0
        )

    if name == "gpt":
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

    raise ValueError("Unknown LLM")
