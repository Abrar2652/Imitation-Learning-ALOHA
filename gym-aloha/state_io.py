import os
import yaml
import json
import numpy as np
import base64
import cv2
import time

SAVE_DIR = "."
os.makedirs(SAVE_DIR, exist_ok=True)

def save_state(env, action=None, filename=None, suffix="", save_dir=SAVE_DIR, ts=None):
    """Save current environment state to file."""
    physics = env.unwrapped._env._physics
    
    # Get timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    if filename is None:
        filename = f"{timestamp}{f'_{suffix}' if suffix else ''}.yaml"
    
    # Get platform transform
    target_id = physics.model.name2id("target", "body")
    platform_pos = physics.model.body_pos[target_id].copy()
    platform_quat = physics.model.body_quat[target_id].copy()
    
    # Get current time with nanoseconds
    current_time = time.time_ns() if ts is None else ts
    
    if action is None:
        action = env.unwrapped.current_action.tolist()
    # Get state information
    state = {
        'qpos': physics.data.qpos.tolist(),
        'qvel': physics.data.qvel.tolist(),
        'current_action': action,
        'platform_transform': {
            'pos': platform_pos.tolist(),
            'quat': platform_quat.tolist()
        },
        'time': current_time,
        'suffix': suffix
    }
    
    # Save to file
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        yaml.dump(state, f)
    
    print(f"Saved state to {filename}")
    return filepath

def load_state(env, filename, save_dir=SAVE_DIR):
    """Load state from file.""" 
    # Reset environment first to ensure proper initialization
    env.reset()
    filepath = os.path.join(save_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"State file not found: {filepath}")
        
    # Load state
    with open(filepath, 'r') as f:
        if filename.endswith('.json'):
            state = json.load(f)
        else:  # yaml
            state = yaml.safe_load(f)

    env.set_state(state)
    return state

def render_state(env):
    """Render current state and return base64 encoded image."""
    image = env.render()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/jpeg;base64,{image_base64}'

def load_states_from_dir(save_dir=SAVE_DIR, pattern=None):
    """Load all saved states and sort them by timestamp.
    
    Args:
        save_dir (str): Directory containing saved states. Defaults to SAVE_DIR.
        pattern (str, optional): If provided, only load files containing this pattern in their name.
    
    Returns:
        list: List of state dictionaries, sorted by timestamp.
    """
    states = []
    
    # Get list of files
    files = os.listdir(save_dir)
    if pattern:
        files = [f for f in files if pattern in f]
    
    # Load each state file
    for filename in files:
        if not filename.endswith(('.yaml', '.json')):
            continue
            
        filepath = os.path.join(save_dir, filename)
        try:
            with open(filepath, 'r') as f:
                if filename.endswith('.json'):
                    state = json.load(f)
                else:  # yaml
                    state = yaml.safe_load(f)
                
                # Add filename to state dict for reference
                state['filename'] = filename
                states.append(state)
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    # Sort by timestamp if available, otherwise by filename
    states.sort(key=lambda x: x.get('time', x['filename']))
    
    if not states:
        print(f"No states found in {save_dir}" + (f" matching pattern '{pattern}'" if pattern else ""))
    else:
        print(f"Loaded {len(states)} states from {save_dir}")
        for state in states:
            print(f"- {state['filename']}" + (f" ({state.get('suffix', '')})" if state.get('suffix') else ""))
    
    return states 
