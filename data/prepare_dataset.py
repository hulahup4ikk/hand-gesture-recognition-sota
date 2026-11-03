"""
Prepare and expand the Hand Gesture dataset for Assignment 2.

Steps:
  1. Verify original Assignment 1 data in data/raw/ (subjects: ivan, dias, tanya, damir).
  2. Download a high-diversity public sample (~350 images per class) from HaGRID
     (384p subset) using parallelized streaming HTTP range requests.
  3. Construct data/split/ with:
     - TEST: Strictly Subject 4 ('damir') from A1 (zero data leakage, true unseen test hand).
     - TRAIN: 80% of A1 subjects (ivan, dias, tanya) + 80% of HaGRID public samples.
     - VAL:   20% of A1 subjects (ivan, dias, tanya) + 20% of HaGRID public samples.

Usage:
  python data/prepare_dataset.py
"""

import os
import sys
import shutil
import random
import struct
import zlib
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force UTF-8 stdout
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

CLASSES = ["like", "palm", "peace"]
HAGRID_CLASS_MAP = {
    "like": "train_val_like",
    "palm": "train_val_palm",
    "peace": "train_val_peace",
}
TEST_SUBJECT_PREFIX = "damir_"
SAMPLES_PER_CLASS = 350
SEED = 42
MAX_WORKERS = 24

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PUBLIC_DIR = BASE_DIR / "data" / "public_sample"
SPLIT_DIR = BASE_DIR / "data" / "split"
HAGRID_ZIP_URL = "https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p/resolve/main/hagrid-sample-30k-384p.zip"


def _download_single_entry(item, cls_dir: Path):
    target_file = cls_dir / f"hagrid_{item['filename']}"
    if target_file.exists() and target_file.stat().st_size > 0:
        return True

    # Read local header (30 bytes) + rough headroom to get extra_len
    # Most ZIP extra_len is 0 or 36 bytes. Let's fetch 30 + fn_len + 128 + comp_size in one go!
    fetch_len = 30 + item["fn_len"] + 128 + item["comp_size"]
    req = urllib.request.Request(
        HAGRID_ZIP_URL,
        headers={"Range": f"bytes={item['local_offset']}-{item['local_offset'] + fetch_len - 1}"}
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        chunk = res.read()

    if chunk[:4] != b"PK\x03\x04":
        raise ValueError(f"Invalid local header magic for {item['filename']}")

    l_fn_len = struct.unpack("<H", chunk[26:28])[0]
    l_extra_len = struct.unpack("<H", chunk[28:30])[0]
    data_start = 30 + l_fn_len + l_extra_len
    c_bytes = chunk[data_start:data_start + item["comp_size"]]

    if item["comp_method"] == 8:
        img_bytes = zlib.decompress(c_bytes, -15)
    elif item["comp_method"] == 0:
        img_bytes = c_bytes
    else:
        raise ValueError(f"Unsupported compression method: {item['comp_method']}")

    with open(target_file, "wb") as f:
        f.write(img_bytes)

    return True


def download_hagrid_subset(num_per_class: int = SAMPLES_PER_CLASS, seed: int = SEED):
    """Download a subset of HaGRID directly from the remote zip using parallel HTTP Range requests."""
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    all_done = True
    for cls in CLASSES:
        cls_dir = PUBLIC_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        existing = [f for f in cls_dir.glob("*.jpg") if f.stat().st_size > 0]
        if len(existing) < num_per_class:
            all_done = False
            break

    if all_done:
        print(f"[OK] Public sample already present in {PUBLIC_DIR} (>={num_per_class} per class). Skipping download.", flush=True)
        return

    print(f"[*] Fetching HaGRID zip central directory from {HAGRID_ZIP_URL}...", flush=True)
    req = urllib.request.Request(HAGRID_ZIP_URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as res:
        total_size = int(res.headers["Content-Length"])

    # Read last 64KB to find End of Central Directory (EOCD)
    tail_len = min(65536, total_size)
    req = urllib.request.Request(
        HAGRID_ZIP_URL,
        headers={"Range": f"bytes={total_size - tail_len}-{total_size - 1}"}
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        tail = res.read()

    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos == -1:
        raise RuntimeError("Could not find EOCD in remote zip archive.")

    eocd = tail[eocd_pos:eocd_pos + 22]
    _, _, _, _, total_entries, cd_size, cd_offset, _ = struct.unpack("<4sHHHHIIH", eocd)
    print(f"[*] Remote zip entries: {total_entries}, CD size: {cd_size / (1024*1024):.2f} MB", flush=True)

    # Read the full Central Directory
    req = urllib.request.Request(
        HAGRID_ZIP_URL,
        headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"}
    )
    with urllib.request.urlopen(req, timeout=60) as res:
        cd_data = res.read()

    # Parse entries for each gesture class
    class_entries = {c: [] for c in CLASSES}
    pos = 0
    while pos < len(cd_data):
        if cd_data[pos:pos + 4] != b"PK\x01\x02":
            break
        fn_len = struct.unpack("<H", cd_data[pos + 28:pos + 30])[0]
        extra_len = struct.unpack("<H", cd_data[pos + 30:pos + 32])[0]
        comment_len = struct.unpack("<H", cd_data[pos + 32:pos + 34])[0]
        local_header_offset = struct.unpack("<I", cd_data[pos + 42:pos + 46])[0]
        comp_size = struct.unpack("<I", cd_data[pos + 20:pos + 24])[0]
        uncomp_size = struct.unpack("<I", cd_data[pos + 24:pos + 28])[0]
        comp_method = struct.unpack("<H", cd_data[pos + 10:pos + 12])[0]

        fn = cd_data[pos + 46:pos + 46 + fn_len].decode("utf-8", errors="ignore")
        parts = fn.strip("/").split("/")

        if len(parts) > 3 and parts[1] == "hagrid_30k" and fn.lower().endswith((".jpg", ".jpeg")):
            sub_name = parts[2]
            for cls, mapped in HAGRID_CLASS_MAP.items():
                if sub_name == mapped:
                    class_entries[cls].append({
                        "filename": parts[-1],
                        "fn_len": fn_len,
                        "local_offset": local_header_offset,
                        "comp_size": comp_size,
                        "uncomp_size": uncomp_size,
                        "comp_method": comp_method,
                    })

        pos += 46 + fn_len + extra_len + comment_len

    # Download selected samples in parallel
    rng = random.Random(seed)
    for cls in CLASSES:
        cls_dir = PUBLIC_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        available = class_entries[cls]
        rng.shuffle(available)
        selected = available[:num_per_class]

        print(f"[*] Downloading {len(selected)} samples for class '{cls}' ({MAX_WORKERS} threads)...", flush=True)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_download_single_entry, item, cls_dir) for item in selected]
            done_count = 0
            for fut in as_completed(futures):
                fut.result()
                done_count += 1
                if done_count % 50 == 0 or done_count == len(selected):
                    print(f"    [{cls}] Downloaded {done_count}/{len(selected)}", flush=True)

    print("[OK] HaGRID subset download complete!", flush=True)


def build_splits(val_ratio: float = 0.20, seed: int = SEED):
    """Assemble final train / val / test sets."""
    print("\n[*] Assembling benchmark splits in data/split/...", flush=True)
    rng = random.Random(seed)

    for split_name in ("train", "val", "test"):
        for cls in CLASSES:
            target_dir = SPLIT_DIR / split_name / cls
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        s: {
            c: {"a1": 0, "hagrid": 0, "total": 0} for c in CLASSES
        } for s in ("train", "val", "test")
    }

    for cls in CLASSES:
        # 1. Process A1 images
        raw_cls = RAW_DIR / cls
        raw_files = [f for f in raw_cls.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        test_a1 = [f for f in raw_files if f.name.lower().startswith(TEST_SUBJECT_PREFIX)]
        trainval_a1 = [f for f in raw_files if not f.name.lower().startswith(TEST_SUBJECT_PREFIX)]

        # Copy test directly (Subject 4 Damir only)
        for f in test_a1:
            shutil.copy2(f, SPLIT_DIR / "test" / cls / f.name)
            stats["test"][cls]["a1"] += 1
            stats["test"][cls]["total"] += 1

        # Split A1 train/val
        rng.shuffle(trainval_a1)
        n_val_a1 = max(1, round(len(trainval_a1) * val_ratio))
        val_a1 = trainval_a1[:n_val_a1]
        train_a1 = trainval_a1[n_val_a1:]

        for f in train_a1:
            shutil.copy2(f, SPLIT_DIR / "train" / cls / f.name)
            stats["train"][cls]["a1"] += 1
            stats["train"][cls]["total"] += 1

        for f in val_a1:
            shutil.copy2(f, SPLIT_DIR / "val" / cls / f.name)
            stats["val"][cls]["a1"] += 1
            stats["val"][cls]["total"] += 1

        # 2. Process HaGRID images
        public_cls = PUBLIC_DIR / cls
        public_files = [f for f in public_cls.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        rng.shuffle(public_files)

        n_val_pub = max(1, round(len(public_files) * val_ratio))
        val_pub = public_files[:n_val_pub]
        train_pub = public_files[n_val_pub:]

        for f in train_pub:
            shutil.copy2(f, SPLIT_DIR / "train" / cls / f.name)
            stats["train"][cls]["hagrid"] += 1
            stats["train"][cls]["total"] += 1

        for f in val_pub:
            shutil.copy2(f, SPLIT_DIR / "val" / cls / f.name)
            stats["val"][cls]["hagrid"] += 1
            stats["val"][cls]["total"] += 1

    # Print Report
    print("=" * 68, flush=True)
    print(f"{'Split':6s} | {'Class':6s} | {'A1 Count':10s} | {'HaGRID Count':14s} | {'Total':6s}", flush=True)
    print("-" * 68, flush=True)
    totals_by_split = {"train": 0, "val": 0, "test": 0}
    for s in ("train", "val", "test"):
        for c in CLASSES:
            a1_c = stats[s][c]["a1"]
            hg_c = stats[s][c]["hagrid"]
            tot_c = stats[s][c]["total"]
            totals_by_split[s] += tot_c
            print(f"{s:6s} | {c:6s} | {a1_c:10d} | {hg_c:14d} | {tot_c:6d}", flush=True)
        print("-" * 68, flush=True)

    print(f"TOTAL: Train={totals_by_split['train']}, Val={totals_by_split['val']}, Test={totals_by_split['test']}", flush=True)
    print(f"Overall Dataset Size: {sum(totals_by_split.values())} images", flush=True)
    print(f"[Verification] Subject 4 test leakage: {stats['test']['like']['hagrid'] + stats['test']['palm']['hagrid'] + stats['test']['peace']['hagrid']} HaGRID in test (must be 0)", flush=True)
    print(f"[Verification] Test set is 100% Subject 4 from A1: {totals_by_split['test']} images.", flush=True)
    print("=" * 68, flush=True)


if __name__ == "__main__":
    download_hagrid_subset(num_per_class=SAMPLES_PER_CLASS)
    build_splits(val_ratio=0.20)
