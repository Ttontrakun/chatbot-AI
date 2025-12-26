# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")

VECTOR_DB_PATH = "./vectorstore"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DEFAULT_LLM = "gpt-4o-mini"
