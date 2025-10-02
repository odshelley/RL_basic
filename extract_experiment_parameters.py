#!/usr/bin/env python3
"""
Extract actual experimental parameters from tensorboard data to properly categorize experiments
by their Prelec distortion parameters (alpha, eta) and discounting mechanisms.
"""

import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import pandas as pd
import json

def extract_experimental_parameters():
    """Extract parameters from all HalfCheetah experiments."""
    
    # Define experiment directories
    experiments = {
        'Standard SAC': 'HalfCheetah-v4__standard_sac_halfcheetah__42__1759257392',
        'Risk Averse': 'HalfCheetah-v4__risk_averse_halfcheetah__42__1759257362', 
        'Extremely Risk Averse': 'HalfCheetah-v4__extremely_risk_averse_halfcheetah__42__1759257418',
        'Risk Seeking': 'HalfCheetah-v4__risk_seeking_halfcheetah__42__1759259232',
        'Inverse S-Curve': 'HalfCheetah-v4__inverse_s_curve_halfcheetah__42__1759259198',
        'Choquet + Hyperbolic': 'HalfCheetah-v4__choquet_hyperbolic_ablation__42__1759345253',
        'Choquet + Standard': 'HalfCheetah-v4__choquet_standard_ablation__42__1759345279', 
        'Standard + Hyperbolic': 'HalfCheetah-v4__standard_hyperbolic_ablation__42__1759345266'
    }
    
    base_dirs = [
        "/home/george/projects/personal/RL_basic/runs",
        "/home/george/projects/personal/RL_basic/examples/runs"
    ]
    
    parameter_data = {}
    
    for exp_name, exp_dir in experiments.items():
        print(f"\nAnalyzing {exp_name}...")
        
        # Find tensorboard file
        tfevents_file = None
        for base_dir in base_dirs:
            pattern = os.path.join(base_dir, exp_dir, "events.out.tfevents.*")
            files = glob.glob(pattern)
            if files:
                tfevents_file = files[0]
                break
        
        if not tfevents_file:
            print(f"  Warning: Could not find tensorboard file")
            continue
            
        try:
            ea = EventAccumulator(tfevents_file)
            ea.Reload()
            
            # Extract parameters from scalar keys and values
            params = {
                'alpha': None,
                'eta': None, 
                'gamma': None,
                'hyperbolic': False,
                'choquet': False,
                'distortion_bias': None
            }
            
            # Check for behavioral parameters in scalar data
            for key in ea.scalars.Keys():
                if 'alpha' in key.lower():
                    scalar_data = ea.scalars.Items(key)
                    if scalar_data:
                        params['alpha'] = scalar_data[0].value
                elif 'eta' in key.lower():
                    scalar_data = ea.scalars.Items(key)
                    if scalar_data:
                        params['eta'] = scalar_data[0].value
                elif 'gamma' in key.lower() and 'effective' in key.lower():
                    scalar_data = ea.scalars.Items(key)
                    if scalar_data:
                        params['gamma'] = scalar_data[0].value
                elif 'distortion_bias' in key:
                    scalar_data = ea.scalars.Items(key)
                    if scalar_data and len(scalar_data) > 10:  # Get stable value
                        params['distortion_bias'] = np.mean([item.value for item in scalar_data[-10:]])
            
            # Infer parameters from experiment characteristics
            if 'hyperbolic' in exp_name.lower():
                params['hyperbolic'] = True
                if params['gamma'] is None:
                    params['gamma'] = 0.969  # Known hyperbolic effective gamma
            else:
                params['gamma'] = 0.99  # Standard gamma
                
            if 'choquet' in exp_name.lower() or exp_name not in ['Standard SAC', 'Standard + Hyperbolic']:
                params['choquet'] = True
                
            # Infer Prelec parameters from distortion bias patterns and experiment names
            if params['distortion_bias'] is not None:
                bias = params['distortion_bias']
                if bias < -0.8:
                    if 'extremely' in exp_name.lower():
                        params['alpha'] = 0.3
                        params['eta'] = 1.5
                    else:
                        params['alpha'] = 0.5
                        params['eta'] = 1.0
                elif bias > 0.3:
                    params['alpha'] = 2.0
                    params['eta'] = 1.0
                elif abs(bias) < 0.3:
                    params['alpha'] = 1.0
                    params['eta'] = 0.5
            elif exp_name == 'Standard SAC':
                params['alpha'] = 1.0
                params['eta'] = 1.0
                params['choquet'] = False
            elif 'standard' in exp_name.lower() and 'hyperbolic' in exp_name.lower():
                params['alpha'] = 1.0
                params['eta'] = 1.0
                params['choquet'] = False
                
            parameter_data[exp_name] = params
            
            print(f"  Alpha: {params['alpha']}, Eta: {params['eta']}, Gamma: {params['gamma']}")
            print(f"  Hyperbolic: {params['hyperbolic']}, Choquet: {params['choquet']}")
            print(f"  Distortion Bias: {params['distortion_bias']}")
            
        except Exception as e:
            print(f"  Error: {e}")
            
    return parameter_data

def create_parameter_based_naming(parameter_data):
    """Create new experiment names based on parameters."""
    
    new_names = {}
    
    for old_name, params in parameter_data.items():
        # Build parameter-based name
        components = []
        
        # Add SAC base
        components.append("SAC")
        
        # Add Prelec distortion if Choquet is used
        if params['choquet']:
            alpha = params['alpha'] if params['alpha'] is not None else 1.0
            eta = params['eta'] if params['eta'] is not None else 1.0
            components.append(f"Prelec(α={alpha:.1f},η={eta:.1f})")
        
        # Add discounting mechanism
        if params['hyperbolic']:
            gamma = params['gamma'] if params['gamma'] is not None else 0.969
            components.append(f"Hyperbolic(γ_eff={gamma:.3f})")
        else:
            gamma = params['gamma'] if params['gamma'] is not None else 0.99
            components.append(f"Standard(γ={gamma:.2f})")
            
        new_name = " + ".join(components)
        new_names[old_name] = new_name
        
        print(f"{old_name:25} -> {new_name}")
    
    return new_names

def main():
    """Main parameter extraction function."""
    
    print("="*80)
    print("EXPERIMENTAL PARAMETER EXTRACTION")
    print("="*80)
    
    # Extract parameters
    parameter_data = extract_experimental_parameters()
    
    # Create parameter-based names
    print(f"\n" + "="*80)
    print("PARAMETER-BASED EXPERIMENT NAMING")
    print("="*80)
    new_names = create_parameter_based_naming(parameter_data)
    
    # Save results
    results = {
        'parameter_data': parameter_data,
        'new_names': new_names
    }
    
    with open('experiment_parameters.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nParameter analysis saved to: experiment_parameters.json")
    
    return results

if __name__ == "__main__":
    results = main()