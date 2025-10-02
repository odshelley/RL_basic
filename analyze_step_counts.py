#!/usr/bin/env python3
"""
Analyze step counts across all experiments to find the minimum common timeframe.
"""

import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import pandas as pd

def get_experiment_step_count(tfevents_file):
    """Get the maximum step count from a tensorboard events file."""
    try:
        ea = EventAccumulator(tfevents_file)
        ea.Reload()
        
        # Try to get charts/episodic_return which is our main metric
        if 'charts/episodic_return' in ea.scalars.Keys():
            scalar_data = ea.scalars.Items('charts/episodic_return')
            if scalar_data:
                max_step = max(item.step for item in scalar_data)
                return max_step, len(scalar_data)
        
        # Fallback to any available scalar
        available_keys = ea.scalars.Keys()
        if available_keys:
            key = list(available_keys)[0]
            scalar_data = ea.scalars.Items(key)
            if scalar_data:
                max_step = max(item.step for item in scalar_data)
                return max_step, len(scalar_data)
                
        return 0, 0
        
    except Exception as e:
        print(f"Error processing {tfevents_file}: {e}")
        return 0, 0

def analyze_all_experiments():
    """Analyze step counts across all experiment directories."""
    
    # Find all tfevents files
    runs_pattern = "/home/george/projects/personal/RL_basic/runs/**/events.out.tfevents.*"
    examples_pattern = "/home/george/projects/personal/RL_basic/examples/runs/**/events.out.tfevents.*"
    
    all_files = glob.glob(runs_pattern, recursive=True) + glob.glob(examples_pattern, recursive=True)
    
    print(f"Found {len(all_files)} tensorboard event files")
    
    experiment_data = []
    
    for tfevents_file in all_files:
        # Extract experiment name from path
        exp_dir = os.path.dirname(tfevents_file)
        exp_name = os.path.basename(exp_dir)
        
        max_step, num_points = get_experiment_step_count(tfevents_file)
        
        # Categorize experiments
        category = "unknown"
        if "ablation" in exp_name:
            category = "ablation"
        elif "halfcheetah" in exp_name.lower():
            category = "halfcheetah"
        elif "pendulum" in exp_name.lower():
            category = "pendulum"
        elif "lunarlander" in exp_name.lower():
            category = "lunarlander"
        elif "hopper" in exp_name.lower():
            category = "hopper"
            
        experiment_data.append({
            'name': exp_name,
            'category': category,
            'max_step': max_step,
            'num_points': num_points,
            'file_path': tfevents_file
        })
        
        print(f"{exp_name[:50]:50} | Steps: {max_step:8} | Points: {num_points:4} | Cat: {category}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(experiment_data)
    
    print("\n" + "="*80)
    print("EXPERIMENT STEP COUNT ANALYSIS")
    print("="*80)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"Total experiments: {len(df)}")
    print(f"Max steps across all: {df['max_step'].max()}")
    print(f"Min steps across all: {df['max_step'].min()}")
    print(f"Mean steps: {df['max_step'].mean():.0f}")
    print(f"Median steps: {df['max_step'].median():.0f}")
    
    # By category
    print(f"\nBy Category:")
    category_stats = df.groupby('category')['max_step'].agg(['count', 'min', 'max', 'mean']).round(0)
    print(category_stats)
    
    # Key experiments for ablation study
    print(f"\nAblation Study Experiments:")
    ablation_experiments = df[df['category'] == 'ablation']
    if not ablation_experiments.empty:
        for _, row in ablation_experiments.iterrows():
            print(f"  {row['name']:60} | Steps: {row['max_step']:8}")
        min_ablation_steps = ablation_experiments['max_step'].min()
        print(f"\nMinimum steps across ablation experiments: {min_ablation_steps}")
    
    # HalfCheetah experiments
    print(f"\nHalfCheetah Experiments:")
    halfcheetah_experiments = df[df['category'] == 'halfcheetah']
    if not halfcheetah_experiments.empty:
        for _, row in halfcheetah_experiments.iterrows():
            print(f"  {row['name']:60} | Steps: {row['max_step']:8}")
        min_halfcheetah_steps = halfcheetah_experiments['max_step'].min()
        print(f"\nMinimum steps across HalfCheetah experiments: {min_halfcheetah_steps}")
    
    # Find minimum common timeframe for analysis
    valid_experiments = df[df['max_step'] > 0]
    if not valid_experiments.empty:
        min_common_steps = valid_experiments['max_step'].min()
        print(f"\n" + "="*50)
        print(f"RECOMMENDED ANALYSIS TIMEFRAME: {min_common_steps} steps")
        print(f"This ensures all experiments have data for the full analysis period")
        print("="*50)
        
        # Save results
        results = {
            'min_common_steps': min_common_steps,
            'total_experiments': len(df),
            'valid_experiments': len(valid_experiments),
            'ablation_min_steps': ablation_experiments['max_step'].min() if not ablation_experiments.empty else 0,
            'halfcheetah_min_steps': halfcheetah_experiments['max_step'].min() if not halfcheetah_experiments.empty else 0,
            'experiment_details': df.to_dict('records')
        }
        
        return results
    else:
        print("No valid experiments found!")
        return None

if __name__ == "__main__":
    results = analyze_all_experiments()
    
    # Save to file for later use
    if results:
        import json
        with open('experiment_step_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to experiment_step_analysis.json")