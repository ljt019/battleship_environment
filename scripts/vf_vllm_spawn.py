import multiprocessing as mp
import sys

# Ensure CUDA initialises cleanly in each worker process
mp.set_start_method("spawn", force=True)

from verifiers.inference.vllm_server import cli_main  # type: ignore

if __name__ == "__main__":
    # Forward CLI args by resetting sys.argv so argparse inside cli_main sees them
    import sys as _sys
    _sys.argv = ["vf_vllm_spawn"] + _sys.argv[1:]
    cli_main() 