"""
최종 YOLO 데이터셋 구성 스크립트

구조:
  train/
    images/  합성 262개 + 기타(미사용) 286개
    labels/
  test/
    images/  only_real 201개 (scene-level)
    labels/

사용법:
  python build_dataset.py
"""

import os
import re
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BASE = os.environ.get("TTAREUNGI_DATA_DIR", "./data")

MATCH_RESULT_JSON  = os.path.join(BASE, "match_result_v3.json")
COMPOSED_DATASET   = os.path.join(BASE, "composed_dataset")   # 합성 262개
ROBOFLOW_DIR       = os.path.join(BASE, "roboflow_downloads")
OUTPUT_DIR         = os.path.join(BASE, "yolo_dataset")

RANDOM_SEED = 42
CLASS_ID    = 0  # basket
# ─────────────────────────────────────────


def safe_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def abs_path(p):
    """상대경로면 BASE 기준 절대경로로 변환"""
    if p and not os.path.isabs(p):
        return os.path.join(BASE, p)
    return p


def find_image(base_name, roboflow_dir):
    """roboflow_downloads에서 base_name으로 시작하는 이미지 찾기"""
    prefix = base_name.lower()
    for root, dirs, files in os.walk(roboflow_dir):
        if os.path.basename(root) != "images":
            continue
        for fname in files:
            if fname.lower().startswith(prefix):
                return os.path.join(root, fname)
    return None


def bboxes_to_yolo(bboxes):
    lines = []
    for bb in bboxes:
        cx = max(0.0, min(1.0, bb["cx"]))
        cy = max(0.0, min(1.0, bb["cy"]))
        w  = max(0.0, min(1.0, bb["w"]))
        h  = max(0.0, min(1.0, bb["h"]))
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("YOLO 데이터셋 구성")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    with open(MATCH_RESULT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_img = os.path.join(OUTPUT_DIR, "train", "images")
    train_lbl = os.path.join(OUTPUT_DIR, "train", "labels")
    test_img  = os.path.join(OUTPUT_DIR, "test",  "images")
    test_lbl  = os.path.join(OUTPUT_DIR, "test",  "labels")
    for d in [train_img, train_lbl, test_img, test_lbl]:
        os.makedirs(d, exist_ok=True)

    train_count = 0
    test_count  = 0

    # ── 1. train: 합성 scene 262개 ──
    print("\n▶ [train] 합성 scene 복사 중...")
    composed_images = Path(COMPOSED_DATASET) / "images"
    composed_labels = Path(COMPOSED_DATASET) / "labels"
    for img_path in sorted(composed_images.glob("*.jpg")):
        lbl_path = composed_labels / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        safe_copy(str(img_path), os.path.join(train_img, img_path.name))
        safe_copy(str(lbl_path), os.path.join(train_lbl, lbl_path.name))
        train_count += 1
    print(f"  → {train_count}개 복사 완료")

    # ── 2. train: 기타(미사용) 286개 ──
    print("\n▶ [train] 기타 미사용 원본 286개 복사 중...")
    tc = data["test_candidates"]
    tc_count = 0
    for base_name, info in tc.items():
        img_path = abs_path(info["image_path"])
        if not img_path or not os.path.exists(img_path):
            img_path = find_image(base_name, ROBOFLOW_DIR)
        if not img_path or not os.path.exists(img_path):
            print(f"  [skip] 이미지 없음: {base_name}")
            continue

        ext      = Path(img_path).suffix
        out_name = f"{base_name}_real{ext}"
        safe_copy(img_path, os.path.join(train_img, out_name))

        # 라벨
        lbl_lines = bboxes_to_yolo(info["bboxes"])
        lbl_path  = os.path.join(train_lbl, f"{base_name}_real.txt")
        with open(lbl_path, "w") as f:
            f.write(lbl_lines)

        tc_count   += 1
        train_count += 1
    print(f"  → {tc_count}개 복사 완료")

    # ── 3. test: only_real 고유 scene ──
    print("\n▶ [test] only_real scene 복사 중...")

    # base_name 기준으로 중복 제거
    scene_map = {}  # base_name → {rf_image, bboxes}
    for split in ["train", "test"]:
        for label in ["empty", "not_empty"]:
            for f in data["only_real_matched"][split][label]["files"]:
                base = f["base_name"]
                if base not in scene_map:
                    rf_img = abs_path(f["rf_image"])
                    scene_map[base] = {
                        "rf_image": rf_img,
                        "bboxes":   f["bboxes"]
                    }

    print(f"  고유 scene 수: {len(scene_map)}개")

    for base_name, info in scene_map.items():
        img_path = info["rf_image"]
        if not img_path or not os.path.exists(img_path):
            img_path = find_image(base_name, ROBOFLOW_DIR)
        if not img_path or not os.path.exists(img_path):
            print(f"  [skip] 이미지 없음: {base_name}")
            continue

        ext      = Path(img_path).suffix
        out_name = f"{base_name}_real{ext}"
        safe_copy(img_path, os.path.join(test_img, out_name))

        lbl_lines = bboxes_to_yolo(info["bboxes"])
        lbl_path  = os.path.join(test_lbl, f"{base_name}_real.txt")
        with open(lbl_path, "w") as f:
            f.write(lbl_lines)

        test_count += 1

    print(f"  → {test_count}개 복사 완료")

    # ── 4. dataset.yaml 생성 ──
    yaml_content = f"""path: {OUTPUT_DIR}
train: train/images
val: test/images
test: test/images

nc: 1
names: ['basket']
"""
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # ── 5. 최종 요약 ──
    total = train_count + test_count
    print(f"\n{'='*60}")
    print(f"✅ 데이터셋 구성 완료!")
    print(f"  train : {train_count}개  ({train_count/total*100:.1f}%)")
    print(f"  test  : {test_count}개   ({test_count/total*100:.1f}%)")
    print(f"  전체  : {total}개")
    print(f"  저장  : {OUTPUT_DIR}")
    print(f"  yaml  : {yaml_path}")
    print(f"\n다음 단계: python train_yolo.py")


if __name__ == "__main__":
    main()