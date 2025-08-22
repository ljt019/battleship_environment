import os

############## Training Config ##############

SCRIPT_PATH = "scripts/sft.py"

CUDA_VISIBLE_DEVICES = "0,1"

#############################################


def main():
    os.system(
        f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch --config-file configs/zero3.yaml \
        --num-processes {CUDA_VISIBLE_DEVICES.count(',') + 1} {SCRIPT_PATH}"
    )


if __name__ == "__main__":
    main()
