import os
import json
import random
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch

from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.admin.environment import env_settings


def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def _decode_video_to_cache(video_path: str, cache_dir: str, expected_n_frames: Optional[int] = None):
    _safe_mkdir(cache_dir)
    existing = sorted([f for f in os.listdir(cache_dir) if f.endswith(".jpg")])

    if expected_n_frames is None:
        if len(existing) > 0:
            return
    else:
        if len(existing) == expected_n_frames:
            return
        for f in existing:
            try:
                os.remove(os.path.join(cache_dir, f))
            except OSError:
                pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_path = os.path.join(cache_dir, f"{idx:08d}.jpg")
        if not cv2.imwrite(out_path, frame):
            cap.release()
            raise RuntimeError(f"Failed to write frame: {out_path}")
        idx += 1

    cap.release()

    if expected_n_frames is not None and idx != expected_n_frames:
        raise RuntimeError(
            f"Decoded frame count mismatch for {video_path}. Decoded={idx}, expected={expected_n_frames}."
        )


def _load_annotations_csv_xywh(annotation_file: str, n_frames: int) -> np.ndarray:
    gt = np.zeros((n_frames, 4), dtype=np.float64)
    if not annotation_file or not os.path.isfile(annotation_file):
        return gt

    data = np.loadtxt(annotation_file, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, 4)

    m = min(n_frames, data.shape[0])
    gt[:m, :] = data[:m, :]
    return gt


class UAVPLDataset(BaseVideoDataset):
    def __init__(
        self,
        split: str = "train",
        dataset_root: Optional[str] = None,
        manifest_path: Optional[str] = None,
        cache_root: Optional[str] = None,
        image_loader=None,
        data_fraction: Optional[float] = None,
        force_6ch: bool = True,   # <--- important for your current model
    ):
        dataset_root = dataset_root or os.getenv("TRAIN_DATASET_ROOT", "/content/my_dataset")
        manifest_path = os.path.join(dataset_root, "metadata", "contestant_manifest.json")
        cache_root = cache_root or os.getenv("TRAIN_CACHE_ROOT", "/tmp/uetrack_uavpl_cache")

        super().__init__("uavpl", dataset_root, image_loader=image_loader)

        self.split = split
        self.dataset_root = dataset_root
        self.manifest_path = manifest_path
        self.cache_root = cache_root
        self.force_6ch = force_6ch

        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)

        if self.split not in manifest:
            raise ValueError(f"Split '{self.split}' not found in manifest. Available: {list(manifest.keys())}")

        self.seq_meta: Dict[str, dict] = manifest[self.split]
        self.seq_ids: List[str] = sorted(list(self.seq_meta.keys()))

        if data_fraction is not None:
            k = max(1, int(len(self.seq_ids) * float(data_fraction)))
            self.seq_ids = random.sample(self.seq_ids, k)

    def get_name(self):
        # This becomes data['dataset'] in the sampler. Make sure your TASK_INDEX has UAVPL or alias it.
        return "uavpl"

    def get_num_sequences(self):
        return len(self.seq_ids)

    def _get_seq_cache_dir(self, seq_id: str) -> str:
        return os.path.join(self.cache_root, seq_id.replace("/", "__"))

    def _ensure_decoded(self, seq_id: str) -> Tuple[int, List[str], np.ndarray]:
        meta = self.seq_meta[seq_id]
        n_frames = int(meta["n_frames"])

        video_path = os.path.join(self.dataset_root, meta["video_path"])
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found for {seq_id}: {video_path}")

        cache_dir = self._get_seq_cache_dir(seq_id)
        _decode_video_to_cache(video_path, cache_dir, expected_n_frames=n_frames)

        frames = [os.path.join(cache_dir, f"{i:08d}.jpg") for i in range(n_frames)]

        ann_rel = meta.get("annotation_path", None)
        ann_path = os.path.join(self.dataset_root, ann_rel) if ann_rel else None
        gt = _load_annotations_csv_xywh(ann_path, n_frames=n_frames)

        return n_frames, frames, gt

    def get_sequence_info(self, seq_id: int):
        seq_name = self.seq_ids[int(seq_id)]
        n_frames, _, gt = self._ensure_decoded(seq_name)

        valid = (gt[:, 2] > 0) & (gt[:, 3] > 0)
        valid_t = torch.from_numpy(valid.astype(np.bool_))
        visible_t = valid_t.clone().to(torch.uint8)
        return {"valid": valid_t, "visible": visible_t}

    def get_frames(self, seq_id: int, frame_ids: List[int], anno=None):
        seq_name = self.seq_ids[int(seq_id)]
        n_frames, frame_paths, gt = self._ensure_decoded(seq_name)

        frames = []
        bboxes = []

        for fid in frame_ids:
            fid_int = int(fid)
            fid_int = max(0, min(fid_int, n_frames - 1))

            im = cv2.imread(frame_paths[fid_int])
            if im is None:
                raise RuntimeError(f"Failed to read frame: {frame_paths[fid_int]}")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            if self.force_6ch:
                # Model expects 6 channels (conv weight [*,6,*,*]).
                # Duplicate RGB to create a 6-channel input.
                im = np.concatenate([im, im], axis=2)  # H,W,6

            frames.append(im)
            bboxes.append(gt[fid_int])

        bbox_t = torch.tensor(np.stack(bboxes, axis=0), dtype=torch.float32)
        anno_dict = {"bbox": bbox_t}
        meta = {"object_class_name": "uavpl"}
        return frames, anno_dict, meta