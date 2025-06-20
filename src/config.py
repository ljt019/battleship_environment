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

MAX_TURNS = 100 
SEED = 82 

# ----------------------------
# General Training Configuration
# ----------------------------

BATCH_SIZE = 1  # Reduced batch size to lower memory footprint
MAX_PROMPT_LENGTH = 1024
MAX_COMPLETION_LENGTH = 7164

# ----------------------------
# GRPO Training Configuration
# ----------------------------

NUM_GRPO_SAMPLES = 2000 
NUM_GRPO_EVAL_SAMPLES = 20

GRPO_GRADIENT_ACCUMULATION_STEPS = 8  

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

NUM_DATASET_SAMPLES = 60  
DATASET_PATH = "datasets/battleship-sft"  
HUB_DATASET_NAME = "ljt019/battleship-sft-new-format"  

DATASET_MODEL_SIZE = "14B"

# ----------------------------
# API Configuration
# ----------------------------

MAX_CONCURRENT_API = 60
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
GRPO_NUM_GENERATIONS = 2  

'''
<result move="a1" value="miss" />
<remaining carrier="1" battleship="1" cruiser="0" submarine="1" destroyer="1" />
<state hits="j9" misses="a1" sunk="d5 e5" unknown="98" />
<grid> 
   a b c d e f g h i j
 1 o ? ? ? ? ? ? ? ? ?
 2 ? ? ? ? ? ? ? ? ? ?
 3 ? ? ? ? ? ? ? ? ? ?
 4 ? ? ? ? ? ? ? ? ? ?
 5 ? ? ? x x ? ? ? ? ?
 6 ? ? ? ? ? ? ? ? ? ?
 7 ? ? ? ? ? ? ? ? ? ?
 8 ? ? ? ? ? ? ? ? ? ?
 9 ? ? ? ? ? ? ? ? ? x
10 ? ? ? ? ? ? ? ? ? ?
</grid>

Next Move:
''' 