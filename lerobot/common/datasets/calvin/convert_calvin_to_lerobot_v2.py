#!/usr/bin/env python
"""
Convert Calvin dataset (training/validation) to a LeRobotDataset v2 using the LeRobotDataset API.
- Load all .npz episode files from both training and validation splits.
- Load the corresponding lang_annotations/auto_lang_ann.npy for each split.
- Map each episode to its language instruction using the 'indx' and 'ann' fields.
- Create a single LeRobot v2 dataset with proper splits and episode organization.
- For each episode:
  - For each timestep, build a frame dict with all available keys.
  - Add each frame to the LeRobotDataset using add_frame.
  - After all frames, call save_episode().
- Output is a LeRobotDataset v2 directory structure with proper splits.
"""
import numpy as np
import argparse
import time
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_FEATURES

def parse_args():
    parser = argparse.ArgumentParser(description='Convert Calvin dataset to LeRobotDataset v2 format')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of threads to use for processing episodes (default: 4)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Number of episodes to process per batch (default: auto-calculate based on total episodes)')
    parser.add_argument('--calvin-root', type=str, default='/data/fywang/Calvin/task_ABCD_D',
                       help='Path to Calvin dataset root directory')
    parser.add_argument('--output-root', type=str, default=None,
                       help='Path to output directory (default: {calvin_root}/lerobot_v2_dataset)')
    parser.add_argument('--repo-id', type=str, default='calvin_task_ABCD_D',
                       help='Repository ID for the LeRobot dataset')
    return parser.parse_args()

CALVIN_ROOT = Path('/data/fywang/Calvin/task_ABCD_D')
OUTPUT_ROOT = CALVIN_ROOT / 'lerobot_v2_dataset'
SPLITS = ['training', 'validation']
ALL_KEYS = [
    'rel_actions', 'robot_obs', 'scene_obs',
    'rgb_static', 'rgb_gripper',
]

# 1. Define features dict for LeRobotDataset
# You may need to adjust dtype/shape for your data!
def get_features_from_sample(sample):
    # Use the same mapping as map_calvin_to_lerobot_keys
    mapping = {
        'scene_obs': 'observation.environment_state',
        'robot_obs': 'observation.state',
        'rgb_static': 'observation.images.top',
        'rgb_gripper': 'observation.images.wrist',
        'rel_actions': 'action',
    }
    features = {}
    for key, arr in sample.items():
        mapped_key = mapping.get(key, key)
        if mapped_key in ['observation.images.top', 'observation.images.wrist']:
            shape = (arr.shape[2], arr.shape[0], arr.shape[1]) if arr.ndim == 3 else arr.shape
            features[mapped_key] = {
                "dtype": "image",
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
        else:
            features[mapped_key] = {
                "dtype": "float64",
                "shape": arr.shape,
                "names": None,
            }
    return {**features, **DEFAULT_FEATURES}

def load_episode_frames(args):
    split_dir, ALL_KEYS, start, end, lang, fps = args
    frames = []
    for t, frame_idx in enumerate(range(start, end)):
        npz_path = split_dir / f"episode_{str(frame_idx).zfill(7)}.npz"
        if not npz_path.exists():
            print(f"Warning: {npz_path} does not exist, skipping.")
            continue
        frame_data = np.load(npz_path, allow_pickle=True)
        frame = {}
        for k in ALL_KEYS:
            if k in frame_data:
                arr = frame_data[k]
                if k in ['rgb_static', 'rgb_gripper'] and arr.ndim == 3:
                    arr = np.transpose(arr, (2, 0, 1))
                frame[k] = arr
        frames.append((frame, lang, t / fps))
    return frames

def map_calvin_to_lerobot_keys(frame):
    """Map Calvin keys to LeRobot keys as per user specification."""
    mapping = {
        'scene_obs': 'observation.environment_state',
        'robot_obs': 'observation.state',
        'rgb_static': 'observation.images.top',
        'rgb_gripper': 'observation.images.wrist',
        'rel_actions': 'action',
    }
    new_frame = {}
    for k, v in frame.items():
        new_key = mapping.get(k, k)
        new_frame[new_key] = v
    return new_frame

def main():
    args = parse_args()

    # Update paths based on command line arguments
    global CALVIN_ROOT, OUTPUT_ROOT
    CALVIN_ROOT = Path(args.calvin_root)
    if args.output_root:
        OUTPUT_ROOT = Path(args.output_root)
    else:
        OUTPUT_ROOT = CALVIN_ROOT / 'lerobot_v2_dataset'

    print(f"Using {args.threads} threads for processing")
    print(f"Calvin root: {CALVIN_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")

    # Start overall timer
    overall_start_time = datetime.now()
    print(f"Started at: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # First pass: count total episodes and get features
    total_episodes_to_process = 0
    all_episodes_data = []

    for split in SPLITS:
        split_dir = CALVIN_ROOT / split
        lang_ann_path = split_dir / 'lang_annotations' / 'auto_lang_ann.npy'
        lang_ann = np.load(lang_ann_path, allow_pickle=True).item()
        ann_list = lang_ann['language']['ann']
        indx_list = lang_ann['info']['indx']

        # Store episode data for processing
        for ep_idx, (start, end) in enumerate(indx_list):
            all_episodes_data.append({
                'split': split,
                'split_dir': split_dir,
                'start': start,
                'end': end,
                'lang': ann_list[ep_idx],
                'original_ep_idx': ep_idx
            })

        total_episodes_to_process += len(indx_list)

    print(f"Total episodes to process: {total_episodes_to_process}")
    print(f"Training episodes: {len([ep for ep in all_episodes_data if ep['split'] == 'training'])}")
    print(f"Validation episodes: {len([ep for ep in all_episodes_data if ep['split'] == 'validation'])}")

    # Get features from first episode
    first_episode = all_episodes_data[0]
    first_npz = first_episode['split_dir'] / f"episode_{str(first_episode['start']).zfill(7)}.npz"
    first_frame = np.load(first_npz, allow_pickle=True)
    features = get_features_from_sample({k: first_frame[k] for k in ALL_KEYS if k in first_frame})

    # Create single LeRobot v2 dataset
    fps = 10  # Calvin is usually 10Hz
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        root=OUTPUT_ROOT,
        use_videos=False,
        robot_type="calvin_robot",
    )

    # Process episodes in batches to avoid memory issues
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = max(1, len(all_episodes_data) // 10)  # Process in 10 batches

    total_batches = (len(all_episodes_data) + batch_size - 1) // batch_size
    total_episodes_processed = 0

    print(f"Processing {len(all_episodes_data)} episodes in batches of {batch_size}")
    print(f"Started at: {overall_start_time.strftime('%H:%M:%S')}")

    for batch_start in range(0, len(all_episodes_data), batch_size):
        batch_start_time = datetime.now()
        batch_end = min(batch_start + batch_size, len(all_episodes_data))
        batch_episodes = all_episodes_data[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1

        print(f"\n--- Batch {batch_num}/{total_batches} (episodes {batch_start+1}-{batch_end}) ---")
        print(f"Batch started at: {batch_start_time.strftime('%H:%M:%S')}")

        # Prepare arguments for multiprocessing (only for this batch)
        episode_args = [
            (ep['split_dir'], ALL_KEYS, ep['start'], ep['end'], ep['lang'], fps)
            for ep in batch_episodes
        ]

        with ProcessPoolExecutor(max_workers=args.threads) as executor:
            for episode_frames in tqdm(executor.map(load_episode_frames, episode_args),
                                     total=len(episode_args),
                                     desc=f"batch {batch_num}"):
                for frame, lang, timestamp in episode_frames:
                    frame = map_calvin_to_lerobot_keys(frame)
                    ds.add_frame(frame, task=lang, timestamp=timestamp)
                ds.save_episode()

        # Update progress
        total_episodes_processed += len(batch_episodes)
        batch_end_time = datetime.now()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()

        print(f"Batch {batch_num} completed at: {batch_end_time.strftime('%H:%M:%S')}")
        print(f"Batch duration: {batch_duration:.1f}s")
        print(f"Progress: {total_episodes_processed}/{total_episodes_to_process} episodes ({total_episodes_processed/total_episodes_to_process*100:.1f}%)")

        # Estimate remaining time
        if total_episodes_processed > 0:
            elapsed_time = (datetime.now() - overall_start_time).total_seconds()
            episodes_per_second = total_episodes_processed / elapsed_time
            remaining_episodes = total_episodes_to_process - total_episodes_processed
            remaining_time = remaining_episodes / episodes_per_second
            eta = datetime.now() + timedelta(seconds=remaining_time)
            print(f"Estimated completion: {eta.strftime('%Y-%m-%d %H:%M:%S')} ({remaining_time/3600:.1f} hours remaining)")

    # Update dataset info with proper splits
    # Calculate split boundaries
    training_episodes = len([ep for ep in all_episodes_data if ep['split'] == 'training'])
    validation_episodes = len([ep for ep in all_episodes_data if ep['split'] == 'validation'])

    # Update the dataset info to include proper splits
    ds.meta.info["splits"] = {
        "train": f"0:{training_episodes}",
        "validation": f"{training_episodes}:{training_episodes + validation_episodes}"
    }

    # Write updated info
    from lerobot.common.datasets.utils import write_info
    write_info(ds.meta.info, OUTPUT_ROOT)

    print("Conversion complete. The dataset is now in LeRobotDataset v2 format.")

    overall_end_time = datetime.now()
    total_duration = (overall_end_time - overall_start_time).total_seconds()
    print(f"\n{'='*80}")
    print(f"CONVERSION COMPLETE!")
    print(f"Started: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {overall_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {total_duration/3600:.1f} hours ({total_duration/60:.1f} minutes)")
    print(f"Total episodes processed: {total_episodes_processed}")
    print(f"Training episodes: {training_episodes}")
    print(f"Validation episodes: {validation_episodes}")
    if total_episodes_processed > 0:
        print(f"Average time per episode: {total_duration/total_episodes_processed:.1f} seconds")
    print("The dataset is now in LeRobotDataset v2 format with proper splits.")

if __name__ == "__main__":
    main()
