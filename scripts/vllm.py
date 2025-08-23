import os

############## vLLM Config ##############

MODEL_NAME = "ljt019/Qwen3-4B-Instruct-bs-sft"

CUDA_VISIBLE_DEVICES = "0,1"

##############################################


def main():
    os.system(
        f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} uv run vf-vllm --model {MODEL_NAME} \
        --data-parallel-size {CUDA_VISIBLE_DEVICES.count(',') + 1} --enforce-eager --disable-log-requests"
    )


if __name__ == "__main__":
    main()
