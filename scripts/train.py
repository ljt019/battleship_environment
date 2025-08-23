import os

############## Training Config ##############

SCRIPT_PATH = "scripts/grpo.py"

CUDA_VISIBLE_DEVICES = "2,3"

#############################################


def main():
    os.system(
        f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --config-file configs/zero3.yaml \
        --num-processes {CUDA_VISIBLE_DEVICES.count(',') + 1} {SCRIPT_PATH}"
    )


if __name__ == "__main__":
    main()
