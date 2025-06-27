from datasets import load_dataset

ds = load_dataset('/data/fywang/Calvin/calvin_debug_dataset/lerobot_v2_dataset', split='train')

print(ds[0])
