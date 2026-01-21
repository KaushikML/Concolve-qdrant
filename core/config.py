import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
        self.text_model_name = os.getenv(
            "TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.image_model_name = os.getenv(
            "IMAGE_MODEL_NAME", "openai/clip-vit-base-patch32"
        )
        self.use_ollama = os.getenv("USE_OLLAMA", "true").lower() == "true"
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.ollama_stream = os.getenv("OLLAMA_STREAM", "true").lower() == "true"
        self.ollama_num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "160"))
        self.ollama_temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
        self.tesseract_cmd = os.getenv("TESSERACT_CMD")
        self.data_dir = os.getenv("DATA_DIR", "data")
        self.sqlite_path = os.getenv("SQLITE_PATH", os.path.join(self.data_dir, "app.db"))
        self.claim_sim_threshold = float(os.getenv("CLAIM_SIM_THRESHOLD", "0.85"))
        self.decay_days = int(os.getenv("DECAY_DAYS", "30"))


settings = Settings()
