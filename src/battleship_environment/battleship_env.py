import verifiers as vf

from battleship_emulator import BattleshipEmulator


class BattleshipEnv(vf.MultiTurnEnv):
    # TODO: Implement the enviroment
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def load_environment(**kwargs):
    """Load and configure the environment."""
    # 1. Load dataset
    dataset = vf.load_example_dataset("gsm8k", split="train")

    # 2. Configure parser
    parser = vf.ThinkParser()

    # 3. Define reward functions
    def correct_answer(completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        return 1.0 if response.strip() == answer.strip() else 0.0

    # 4. Create rubric
    rubric = vf.Rubric(
        funcs=[correct_answer, parser.get_format_reward_func()], weights=[1.0, 0.2]
    )

    # 5. Return configured environment
    return vf.BattleshipEnv(
        dataset=dataset,
        system_prompt="Think step-by-step, then give your answer.",
        parser=parser,
        rubric=rubric,
        **kwargs,  # Pass through additional arguments
    )
