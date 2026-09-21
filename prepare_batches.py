"""
match_result_v3.json + v4_non_empty + Roboflow 원본 이미지를 이용해서
match_merge.py가 요구하는 v5_batches 폴더 구조를 생성하는 스크립트
"""

import os, re, json, shutil
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIG (절대경로)
# ─────────────────────────────────────────
BASE = os.environ.get("TTAREUNGI_DATA_DIR", "./data")

MATCH_RESULT_JSON = os.path.join(BASE, "match_result_v3.json")
SYNTHETIC_DIR     = os.path.join(BASE, r"final_training_version\final_training_v4\v4_non_empty")
ROBOFLOW_DIR      = os.path.join(BASE, "roboflow_downloads")
V5_BATCHES_DIR    = os.path.join(BASE, "v5_batches")
# ─────────────────────────────────────────


def safe_imread(path):
    n = np.fromfile(path, np.uint8)
    return cv2.imdecode(n, cv2.IMREAD_COLOR)


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = max(0,      int((cx - w / 2) * img_w))
    y1 = max(0,      int((cy - h / 2) * img_h))
    x2 = min(img_w,  int((cx + w / 2) * img_w))
    y2 = min(img_h,  int((cy + h / 2) * img_h))
    return [x1, y1, x2, y2]


def get_image_size(image_path):
    img = safe_imread(image_path)
    if img is None:
        return None, None
    return img.shape[1], img.shape[0]  # W, H


def extract_rf_hash(image_path):
    m = re.search(r'\.rf\.([a-f0-9]+)', Path(image_path).stem, re.IGNORECASE)
    return m.group(1) if m else None


def main():
    print("=" * 60)
    print("v5_batches 폴더 구조 생성")
    print("=" * 60)

    # 1. JSON 로드
    with open(MATCH_RESULT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    synth_matched = data["synthetic_matched"]
    print(f"\n합성 매칭 파일 수: {len(synth_matched)}개")

    # 2. base_name 기준으로 그룹화
    groups = defaultdict(lambda: {
        "image_path": None,
        "bboxes":     [],
        "crop_files": []   # (filename, src_path)
    })

    synth_dir = Path(SYNTHETIC_DIR)
    for filename, info in synth_matched.items():
        base = info["base_name"]
        groups[base]["image_path"] = info["image_path"]
        groups[base]["bboxes"]     = info["bboxes"]
        src_path = synth_dir / filename
        if src_path.exists():
            groups[base]["crop_files"].append((filename, str(src_path)))
        else:
            print(f"  [경고] 파일 없음: {src_path}")

    print(f"원본 이미지 그룹 수: {len(groups)}개\n")

    # 3. v5_batches 구조 생성
    os.makedirs(V5_BATCHES_DIR, exist_ok=True)
    success, fail = 0, 0

    for base_name, info in groups.items():
        img_path = info["image_path"]
        bboxes   = info["bboxes"]
        crops    = info["crop_files"]

        # 경로가 상대경로면 절대경로로 변환
        if img_path and not os.path.isabs(img_path):
            img_path = os.path.join(BASE, img_path)

        if not img_path or not os.path.exists(img_path):
            print(f"  [skip] 원본 이미지 없음: {base_name} → {img_path}")
            fail += 1
            continue

        if not crops:
            print(f"  [skip] crop 파일 없음: {base_name}")
            fail += 1
            continue

        # 이미지 크기
        img_w, img_h = get_image_size(img_path)
        if img_w is None:
            print(f"  [skip] 이미지 읽기 실패: {img_path}")
            fail += 1
            continue

        # 배치 폴더명
        rf_hash = extract_rf_hash(img_path)
        batch_name = f"{base_name}_jpg.rf.{rf_hash}" if rf_hash else base_name

        batch_dir = Path(V5_BATCHES_DIR) / batch_name
        crops_dir = batch_dir / "crops"
        edits_dir = batch_dir / "edits_in"
        meta_dir  = batch_dir / "meta"
        for d in [crops_dir, edits_dir, meta_dir]:
            os.makedirs(d, exist_ok=True)

        # 4. crop 파일 복사
        for filename, src_path in crops:
            m = re.search(r'_bbox(\d+)', Path(filename).stem, re.IGNORECASE)
            bbox_id = int(m.group(1)) if m else 0

            # crops/ → {base_name}_bbox{N}.png (manifest 생성용)
            shutil.copy2(src_path, str(crops_dir / f"{base_name}_bbox{bbox_id}.png"))
            # edits_in/ → 원본 파일명 그대로
            shutil.copy2(src_path, str(edits_dir / filename))

        # 5. meta json 생성
        meta_bboxes = []
        for i, bb in enumerate(bboxes):
            xyxy = yolo_to_xyxy(bb["cx"], bb["cy"], bb["w"], bb["h"], img_w, img_h)
            meta_bboxes.append({
                "bbox_id": i,
                "xyxy":    xyxy
            })

        meta = {
            "image_id":       base_name,
            "batch_name":     batch_name,
            "original_image": img_path,
            "image_size":     [img_w, img_h],
            "bboxes":         meta_bboxes
        }
        with open(str(meta_dir / f"{batch_name}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        success += 1
        print(f"  ✅ {batch_name}  (crops:{len(crops)}개, bboxes:{len(meta_bboxes)}개)")

    print(f"\n{'='*60}")
    print(f"완료: 성공 {success}개 / 실패 {fail}개")
    print(f"출력 폴더: {V5_BATCHES_DIR}")
    print(f"\n다음 단계: python match_merge.py")


if __name__ == "__main__":
    main()