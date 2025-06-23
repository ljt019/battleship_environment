import multiprocessing as mp

mp.set_start_method("spawn", force=True)

from verifiers.inference.vllm_server import cli_main

if __name__ == "__main__":
    import sys as _sys
    _sys.argv = ["vf_vllm_spawn"] + _sys.argv[1:]
    cli_main() 