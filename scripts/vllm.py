import os

############## vLLM Config ##############

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# MODEL_NAME = "ljt019/Qwen3-4B-Instruct-bs-sft-0825" # for grpo

CUDA_VISIBLE_DEVICES = "0"

##############################################


def main():
    os.system(
        f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} uv run vf-vllm --model {MODEL_NAME} \
        --data-parallel-size {CUDA_VISIBLE_DEVICES.count(',') + 1} --enforce-eager --disable-log-requests"
    )


if __name__ == "__main__":
    main()
