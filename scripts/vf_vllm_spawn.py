import multiprocessing as mp
import sys

# Ensure CUDA initialises cleanly in each worker process
mp.set_start_method("spawn", force=True)

from verifiers.inference.vllm_server import cli  # type: ignore

if __name__ == "__main__":
    # Forward CLI arguments to the real vf-vllm entry-point
    # sys.argv[0] is this script, so pass the rest
    cli(sys.argv[1:]) 