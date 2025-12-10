import sys
import os
import argparse
import yaml
import numpy as np
import imageio.v2 as imageio
import datetime
from pathlib import Path
from tqdm import tqdm

LIBERO_PRO_ROOT = '/mnt/mnt/public/hyx/LIBERO-PRO'

LIBERO_LIB_PATH = LIBERO_PRO_ROOT

if LIBERO_PRO_ROOT not in sys.path:
    sys.path.insert(0, LIBERO_PRO_ROOT)
if LIBERO_LIB_PATH not in sys.path:
    sys.path.insert(1, LIBERO_LIB_PATH)

try:
    import perturbation
    from liberopro.liberopro.envs import OffScreenRenderEnv
except ImportError as e:
    print(f"Import failed: {e}")
    print(f"Please check these paths are correct:\nLIBERO_PRO_ROOT={LIBERO_PRO_ROOT}\nLIBERO_LIB_PATH={LIBERO_LIB_PATH}")
    exit(1)

EPISODE_LENGTH = 200
FPS = 20
RESOLUTION = (512, 512)
OUTPUT_DIR = Path("./libero_pro_output")

def fix_config_paths(config, root_dir):
    """
    Helper: convert relative paths in YAML to absolute paths.
    perturbation.py depends on these paths being real/absolute.
    """
    if "script_path" in config:
        rel_path = config["script_path"].lstrip("./")
        config["script_path"] = os.path.join(root_dir, rel_path)

    if "init_file_dir" in config:
        rel_path = config["init_file_dir"].lstrip("./")
        config["init_file_dir"] = os.path.join(root_dir, rel_path)

    if "ood_task_configs" in config:
        for key, path in config["ood_task_configs"].items():
            rel_path = path.lstrip("./")
            config["ood_task_configs"][key] = os.path.join(root_dir, rel_path)
            
    return config

def get_perturbed_bddl_path(original_suite, task_name, perturb_type, eval_config):
    """
    1. If requesting the original task, return it directly.
    2. If requesting a perturbed task, try to locate an existing perturbed folder.
    3. If the folder does not exist, construct a config and call perturbation.create_env to generate it.
    """
    
    base_bddl_dir = os.path.join(LIBERO_LIB_PATH, "libero","libero", "bddl_files")
    source_dir = os.path.join(base_bddl_dir, original_suite)
    
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Original task suite not found: {source_dir}")

    type_map = {
        "original": None,
        "object": "use_object",
        "spatial": "use_swap",      
        "language": "use_language",
        "task": "use_task",
        "env": "use_environment"
    }
    
    suffix_map = {
        "use_object": "object", 
        "use_swap": "swap", 
        "use_language": "lan", 
        "use_task": "task",
        "use_environment": "env"
    }

    if perturb_type == "original":
        target_path = os.path.join(source_dir, f"{task_name}.bddl")
        return Path(target_path)

    config_key = type_map[perturb_type]
    
    expected_suffix = suffix_map.get(config_key, "temp")
    target_suite_name = f"{original_suite}_{expected_suffix}"
    target_dir = os.path.join(base_bddl_dir, target_suite_name)
    
    temp_suite_name = f"{original_suite}_temp"
    temp_dir = os.path.join(base_bddl_dir, temp_suite_name)

    final_dir = target_dir
    if not os.path.exists(target_dir):
        print(f"Standard directory {target_suite_name} does not exist, preparing to generate temporary environment...")
        final_dir = temp_dir  # final output will be placed in the _temp dir

        run_config = eval_config.copy()
        
        run_config["bddl_files_path"] = source_dir
        run_config["task_suite_name"] = original_suite
        
        for k in type_map.values():
            if k: run_config[k] = False
        run_config[config_key] = True
        
        run_config["seed"] = 10000 

    print(f"⚙️ Calling perturbation.create_env, enabling {config_key}...")
    perturbation.create_env(configs=run_config)
    print("Environment generation complete.")
    
    target_bddl_path = os.path.join(final_dir, f"{task_name}.bddl")
    
    if not os.path.exists(target_bddl_path):
        fallback_path = os.path.join(temp_dir, f"{task_name}.bddl")
        if os.path.exists(fallback_path):
            return Path(fallback_path)
        raise FileNotFoundError(f"Could not find generated BDDL file: {target_bddl_path}")

    return Path(target_bddl_path)

def setup_environment(bddl_file_path: Path, resolution: tuple):
    print(f"Loading BDDL: {bddl_file_path}")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file_path),
        camera_names=["agentview", "robot0_eye_in_hand"],
        has_renderer=False,
        has_offscreen_renderer=True,
        control_freq=FPS,
        render_camera="agentview",
        camera_heights=resolution[1],
        camera_widths=resolution[0],
    )
    env.seed(np.random.randint(0, 1000))
    return env

def run_random_episode(env, episode_length, output_path, task_name):
    print(f"\n--- Starting random simulation: {task_name} ---")
    try:
        action_spec = env.env.action_spec
    except AttributeError:
        action_spec = env.action_spec
    low, high = action_spec[0], action_spec[1]
    
    frames = []
    obs = env.reset()
    if obs.get('agentview_image') is not None:
        frames.append(np.flip(obs.get('agentview_image'), axis=0))

    for _ in tqdm(range(episode_length), desc="Running Steps"):
        action = np.random.uniform(low=low, high=high)
        obs, _, done, _ = env.step(action)
        if obs.get('agentview_image') is not None:
            frames.append(np.flip(obs.get('agentview_image'), axis=0))
        if done: break

    output_dir_path = Path(output_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    save_path = output_dir_path / f"{task_name}_{timestamp}.mp4"
    
    print(f"Saving video...")
    imageio.mimsave(str(save_path), frames, fps=FPS, quality=9)
    print(f"Video saved to: {save_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="LIBERO-Pro evaluation generator")
    parser.add_argument("--suite", type=str, default="libero_spatial", help="Source task suite name (e.g. libero_10, libero_goal)")
    parser.add_argument("--task_name", type=str, required=True, help="Task name (without .bddl)")
    parser.add_argument("--type", type=str, default="original", 
                        choices=["original", "object", "spatial", "language", "task", "env"],
                        help="Perturbation type")
    args = parser.parse_args()

    config_path = os.path.join(LIBERO_PRO_ROOT, "evaluation_config.yaml")
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    eval_config = fix_config_paths(raw_config, LIBERO_PRO_ROOT)

    try:
        bddl_path = get_perturbed_bddl_path(args.suite, args.task_name, args.type, eval_config)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return

    env = None
    try:
        env = setup_environment(bddl_path, RESOLUTION)
        run_random_episode(env, EPISODE_LENGTH, str(OUTPUT_DIR), f"{args.suite}_{args.type}_{args.task_name}")
    except Exception as e:
        print(f"Simulation runtime error: {e}")
    finally:
        if env: env.close()

if __name__ == "__main__":
    main()