# Model Configuration
MODEL_SIZE = "1.7B"
BASE_MODEL_NAME = f"Qwen/Qwen3-{MODEL_SIZE}"
SFT_MODEL_NAME = f"ljt019/Qwen3-{MODEL_SIZE}-Battleship-SFT"
GRPO_MODEL_NAME = f"ljt019/Qwen3-{MODEL_SIZE}-Battleship-GRPO"

# Training Run Names
SFT_RUN_NAME = f"battleship-sft-{MODEL_SIZE}"
GRPO_RUN_NAME = f"battleship-grpo-{MODEL_SIZE}"

# Environment Configuration  
MAX_TURNS = 45
NUM_SAMPLES = 2000
NUM_EVAL_SAMPLES = 100
SEED = 42

# Dataset Configuration
NUM_DATASET_SAMPLES = 500
NUM_DATASET_EVAL_SAMPLES = 500  
MAX_CONCURRENT = 50 
MAX_TOKENS = 8192

# Training Configuration
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 1
MAX_LENGTH = 8192

# API Configuration
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "token-abc123"

# LM Studio Configuration
LMSTUDIO_BASE_URL = "http://172.21.160.1:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL = "qwen3-8b" 

# Paths
DATASET_PATH = "datasets/battleship-sft"
HUB_DATASET_NAME = "ljt019/battleship-sft"
SFT_OUTPUT_DIR = "sft-battleship" 