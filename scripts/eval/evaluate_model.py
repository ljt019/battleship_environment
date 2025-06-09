import os
import argparse
from openai import OpenAI
import sys

# Add the scripts directory to the path to import battleship modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import verifiers as vf
from battleship_env import BattleshipEnv

vf_env = BattleshipEnv(
    num_samples=2000,
    num_eval_samples=100,
    seed=42
)

def main(api: str, num_samples: int, max_tokens: int, save_dataset: bool = False):
    if api == "vllm":
        base_url = "http://localhost:8000/v1"
        api_key = "token-abc123"
        model_name = "Qwen/Qwen3-1.7B"
        client = OpenAI(base_url=base_url, api_key=api_key)
    elif api == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = "gpt-4o-mini"
        client = OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Invalid API: {api}")
    
    sampling_args = {
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    
    results = vf_env.evaluate(
        client=client,
        model=model_name,
        sampling_args=sampling_args,
        num_samples=num_samples
    )

    print('--- Example ---')
    print('Prompt: ', results['prompt'][0][:200] + '...' if len(results['prompt'][0]) > 200 else results['prompt'][0])
    print('Completion: ', results['completion'][0][:200] + '...' if len(results['completion'][0]) > 200 else results['completion'][0])
    print('Answer: ', results['answer'][0])
    
    print("--- Rewards ---")
    for k, v in results.items():
        if 'reward' in k:
            avg_reward = sum(v) / len(v) if v else 0
            print(k, '-', avg_reward)
    
    # Calculate win rate
    win_rate = sum(1 for r in results.get('rewards', []) if r > 0.5) / len(results.get('rewards', [1])) if results.get('rewards') else 0
    print(f"Win rate: {win_rate:.1%}")
    
    if save_dataset:
        dataset = vf_env.make_dataset(results)
        dataset = dataset.sort("rewards", reverse=True).select(range(len(dataset) // 2))
        dataset.save_to_disk(f"outputs/battleship_eval_{api}_{model_name.replace('/', '_')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", "-a", type=str, default="vllm")
    parser.add_argument("--num-samples", "-n", type=int, default=20)
    parser.add_argument("--max-tokens", "-t", type=int, default=512)
    parser.add_argument("--save-dataset", "-s", action="store_true", default=False)
    args = parser.parse_args()
    
    if args.api == "vllm" and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy"
    
    main(args.api, args.num_samples, args.max_tokens, args.save_dataset)
