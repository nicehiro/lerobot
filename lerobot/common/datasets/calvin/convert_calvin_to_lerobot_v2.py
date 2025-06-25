#!/usr/bin/env python
"""
Convert Calvin dataset (training/validation) to a LeRobotDataset using the LeRobotDataset API.
- For each split (training, validation):
  - Load all .npz episode files.
  - Load the corresponding lang_annotations/auto_lang_ann.npy.
  - Map each episode to its language instruction using the 'indx' and 'ann' fields.
  - For each episode:
    - For each timestep, build a frame dict with all available keys.
    - Add each frame to the LeRobotDataset using add_frame.
    - After all frames, call save_episode().
- Output is a LeRobotDataset directory structure.
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_FEATURES

CALVIN_ROOT = Path('/data/fywang/Calvin/calvin_debug_dataset')
OUTPUT_ROOT = CALVIN_ROOT / 'lerobot_v2_export'
SPLITS = ['training', 'validation']
ALL_KEYS = [
    'actions', 'rel_actions', 'robot_obs', 'scene_obs',
    'rgb_static', 'rgb_gripper',
]

# 1. Define features dict for LeRobotDataset
# You may need to adjust dtype/shape for your data!
def get_features_from_sample(sample):
    features = {}
    for key, arr in sample.items():
        if key in ['rgb_static', 'rgb_gripper']:
            shape = (arr.shape[2], arr.shape[0], arr.shape[1]) if arr.ndim == 3 else arr.shape
            features[key] = {
                "dtype": "image",
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
        else:
            features[key] = {
                "dtype": "float64",
                "shape": arr.shape,
                "names": None,
            }
    return {**features, **DEFAULT_FEATURES}

# 2. Create LeRobotDataset
def create_lerobot_dataset(output_dir, repo_id, fps, features):
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        use_videos=False,
    )

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

for split in SPLITS:
    print(f"Processing split: {split}")
    split_dir = CALVIN_ROOT / split
    lang_ann_path = split_dir / 'lang_annotations' / 'auto_lang_ann.npy'
    lang_ann = np.load(lang_ann_path, allow_pickle=True).item()
    ann_list = lang_ann['language']['ann']
    indx_list = lang_ann['info']['indx']

    first_start, _ = indx_list[0]
    first_npz = split_dir / f"episode_{str(first_start).zfill(7)}.npz"
    first_frame = np.load(first_npz, allow_pickle=True)
    features = get_features_from_sample({k: first_frame[k] for k in ALL_KEYS if k in first_frame})

    repo_id = f"calvin_debug_{split}"
    out_split_dir = OUTPUT_ROOT / split
    fps = 10  # Calvin is usually 10Hz
    ds = create_lerobot_dataset(out_split_dir, repo_id, fps, features)

    # Prepare arguments for multiprocessing
    episode_args = [
        (split_dir, ALL_KEYS, start, end, ann_list[ep_idx], fps)
        for ep_idx, (start, end) in enumerate(indx_list)
    ]

    with ProcessPoolExecutor() as executor:
        for episode_frames in tqdm(executor.map(load_episode_frames, episode_args), total=len(episode_args), desc=f"{split} episodes"):
            for frame, lang, timestamp in episode_frames:
                ds.add_frame(frame, task=lang, timestamp=timestamp)
            ds.save_episode()

print("Conversion complete. The dataset is now in LeRobotDataset format.")
