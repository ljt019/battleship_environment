import os

# ----------------------------
# General Model Configuration
# ----------------------------

MODEL_SIZE = "1.7B"
BASE_MODEL_NAME = f"willcb/Qwen3-{MODEL_SIZE}"  
SFT_MODEL_NAME = f"ljt019/Qwen3-{MODEL_SIZE}-Battleship-SFT" 
GRPO_MODEL_NAME = f"ljt019/Qwen3-{MODEL_SIZE}-Battleship-GRPO"  

# ----------------------------
# WANDB Run Names
# ----------------------------

SFT_RUN_NAME = f"battleship-sft-{MODEL_SIZE}"  
GRPO_RUN_NAME = f"battleship-grpo-{MODEL_SIZE}"

# ----------------------------
# Battleship Game Environment
# ----------------------------

MAX_TURNS = 50  
SEED = 82 

# ----------------------------
# General Training Configuration
# ----------------------------

BATCH_SIZE = 2  # Batch size per device
MAX_PROMPT_LENGTH = 1024
MAX_COMPLETION_LENGTH = 8192

# ----------------------------
# GRPO Training Configuration
# ----------------------------

NUM_GRPO_SAMPLES = 2000 
NUM_GRPO_EVAL_SAMPLES = 20

GRPO_GRADIENT_ACCUMULATION_STEPS = 4  

# ----------------------------
# SFT Training Configuration
# ----------------------------

LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3 

SFT_GRADIENT_ACCUMULATION_STEPS = 1  
SFT_OUTPUT_DIR = "sft-battleship" 

# ----------------------------
# SFT Dataset Generation
# ----------------------------

NUM_DATASET_SAMPLES = 40  
DATASET_PATH = "datasets/battleship-sft"  
HUB_DATASET_NAME = "ljt019/battleship-sft-reformat"  

DATASET_MODEL_SIZE = "14B"

# ----------------------------
# API Configuration
# ----------------------------

MAX_CONCURRENT_API = 8
MAX_TOKENS_API = 8192   

# LM Studio
LMSTUDIO_BASE_URL = "http://172.21.160.1:1234/v1"  
LMSTUDIO_API_KEY = "lm-studio" 
LMSTUDIO_MODEL = "qwen3-8b"  

# OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "deepseek/deepseek-chat-v3-0324"

# vLLM
VLLM_BASE_URL = "http://localhost:8000/v1" 
VLLM_API_KEY = "token-abc123"  

# ----------------------------
# GRPO generation settings
# ----------------------------

# Number of completions ("generations") to produce per prompt during GRPO training.
# This value is used by both the async batch generator and the Trainer when
# reshaping reward tensors – keep it in one place to avoid mismatches.
GRPO_NUM_GENERATIONS = 4  