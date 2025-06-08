"""
Dataset analysis for battleship training data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from datasets import Dataset
import re
import random
from collections import Counter

def load_dataset():
    dataset_path = "datasets/battleship_rlvr_qwen3_dataset"
    
    try:
        dataset = Dataset.load_from_disk(dataset_path)
        print(f"Loaded dataset from {dataset_path}")
        print(f"Size: {len(dataset)} examples")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

def check_move_formats(dataset):
    print("\nMove format validation:")
    
    # Battleship coordinates: a-j (columns) + 1-10 (rows)
    valid_pattern = re.compile(r'^[a-j]([1-9]|10)$')
    valid_moves = 0
    invalid_examples = []
    
    for i, example in enumerate(dataset):
        move = example['answer']
        if valid_pattern.match(move):
            valid_moves += 1
        else:
            invalid_examples.append(f"Example {i}: '{move}'")
            if len(invalid_examples) <= 5:
                print(f"  Invalid: {move}")
    
    format_rate = valid_moves / len(dataset)
    print(f"  Valid: {valid_moves}/{len(dataset)} ({format_rate:.1%})")
    
    return format_rate >= 0.99

def analyze_game_stages(dataset):
    print("\nGame stage distribution:")
    
    stage_counts = {'early': 0, 'mid': 0, 'late': 0}
    move_counts = []
    
    for example in dataset:
        question = example['question']
        
        # Count moves by board state markers
        hits = question.count('[x]')
        misses = question.count('[o]')
        sunk = question.count('[s]')
        total_moves = hits + misses + sunk
        move_counts.append(total_moves)
        
        # Stage classification based on typical game progression
        if total_moves <= 15:
            stage_counts['early'] += 1
        elif total_moves <= 40:
            stage_counts['mid'] += 1
        else:
            stage_counts['late'] += 1
    
    for stage, count in stage_counts.items():
        percentage = count / len(dataset) * 100
        print(f"  {stage}: {count} ({percentage:.1f}%)")
    
    print(f"  Move stats - avg: {sum(move_counts) / len(move_counts):.1f}, "
          f"range: {min(move_counts)}-{max(move_counts)}")
    
    # Check for balanced distribution across game stages
    early_pct = stage_counts['early'] / len(dataset)
    mid_pct = stage_counts['mid'] / len(dataset)
    late_pct = stage_counts['late'] / len(dataset)
    
    balanced = all(0.25 <= pct <= 0.45 for pct in [early_pct, mid_pct, late_pct])
    return balanced

def check_strategic_patterns(dataset):
    print("\nStrategy patterns (sample analysis):")
    
    sample_size = min(20, len(dataset))
    sample_indices = random.sample(range(len(dataset)), sample_size)
    
    adjacency_cases = 0
    hunt_cases = 0
    center_bias_count = 0
    
    # Center squares are statistically better for hunting
    center_squares = ['d4', 'd5', 'd6', 'd7', 'e4', 'e5', 'e6', 'e7', 
                     'f4', 'f5', 'f6', 'f7', 'g4', 'g5', 'g6', 'g7']
    
    for i in sample_indices:
        example = dataset[i]
        question = example['question']
        move = example['answer']
        
        has_hits = '[x]' in question
        
        if has_hits:
            adjacency_cases += 1  # Should target adjacent to hits
        else:
            hunt_cases += 1  # Should prefer center squares
            if move in center_squares:
                center_bias_count += 1
    
    print(f"  Target mode: {adjacency_cases}/{sample_size}")
    print(f"  Hunt mode: {hunt_cases}/{sample_size}")
    if hunt_cases > 0:
        center_rate = center_bias_count / hunt_cases
        print(f"  Center preference: {center_bias_count}/{hunt_cases} ({center_rate:.1%})")
    
    return True

def analyze_move_distribution(dataset):
    print("\nMove distribution:")
    
    move_counts = Counter()
    for example in dataset:
        move_counts[example['answer']] += 1
    
    total_examples = len(dataset)
    most_common = move_counts.most_common(10)
    
    print("  Top moves:")
    for move, count in most_common[:5]:
        percentage = count / total_examples * 100
        print(f"    {move}: {count} ({percentage:.1f}%)")
    
    # Flag if any single move appears too frequently
    max_count = most_common[0][1]
    max_percentage = max_count / total_examples
    
    if max_percentage > 0.08:  # 8% threshold for over-representation
        print(f"  Warning: '{most_common[0][0]}' over-represented ({max_percentage:.1%})")
    
    return max_percentage <= 0.08

def check_format_consistency(dataset):
    print("\nFormat consistency:")
    
    expected_start = "Given this battleship board state, what is the best next move?"
    consistent_format = 0
    complete_boards = 0
    
    for example in dataset:
        question = example['question']
        
        if question.startswith(expected_start):
            consistent_format += 1
            
        # Check for required board components
        if all(marker in question for marker in ['[?]', 'Remaining ships:', 'Turn history:']):
            complete_boards += 1
    
    format_rate = consistent_format / len(dataset)
    completeness_rate = complete_boards / len(dataset)
    
    print(f"  Question format: {consistent_format}/{len(dataset)} ({format_rate:.1%})")
    print(f"  Board completeness: {complete_boards}/{len(dataset)} ({completeness_rate:.1%})")
    
    return format_rate >= 0.99 and completeness_rate >= 0.99

def show_samples(dataset, num_samples=2):
    print(f"\nSample examples:")
    print("-" * 50)
    
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for i, idx in enumerate(sample_indices):
        example = dataset[idx]
        print(f"\nExample {i+1}:")
        question_preview = example['question'][:300] + "..." if len(example['question']) > 300 else example['question']
        print(question_preview)
        print(f"Answer: {example['answer']}")

def run_analysis():
    print("Dataset Analysis")
    print("=" * 40)
    
    dataset = load_dataset()
    if not dataset:
        return False
    
    # Run validation checks
    checks = [
        check_move_formats(dataset),
        analyze_game_stages(dataset),
        check_strategic_patterns(dataset),
        analyze_move_distribution(dataset),
        check_format_consistency(dataset)
    ]
    
    show_samples(dataset)
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\nSummary: {passed}/{total} checks passed")
    
    if passed == total:
        print("Status: Ready for training")
        return True
    else:
        print("Status: Issues found, review needed")
        return False

if __name__ == "__main__":
    success = run_analysis()
    exit(0 if success else 1) 