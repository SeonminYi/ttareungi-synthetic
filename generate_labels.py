"""
composed_output_v2/ 이미지들의 YOLO 라벨 생성
meta/ 폴더의 xyxy 좌표 → YOLO 정규화 좌표(cx cy w h) 변환
"""

import os
import re
import json
import shutil
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BASE = os.path.join(os.environ.get("TTAREUNGI_DATA_DIR", "./data"), "yolo_retraining")

COMPOSED_DIR  = os.path.join(BASE, "composed_output_v2")
V5_BATCHES    = os.path.join(BASE, "v5_batches")
DATASET_DIR   = os.path.join(BASE, "bicycle-basket-4")

CLASS_ID = 0  # basket
# ─────────────────────────────────────────


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def find_meta(image_id: str) -> str | None:
    """v5_batches에서 image_id에 해당하는 meta json 찾기"""
    v5 = Path(V5_BATCHES)
    for batch_dir in v5.iterdir():
        if not batch_dir.is_dir():
            continue
        # batch_name: 526_m_2_jpg.rf.xxx
        # image_id:   526_m_2
        batch_base = re.sub(
            r'_[a-zA-Z]+\.rf\.[a-f0-9]+$', '',
            batch_dir.name, flags=re.IGNORECASE
        ).lower()
        if batch_base == image_id.lower():
            meta_path = batch_dir / "meta" / f"{batch_dir.name}.json"
            if meta_path.exists():
                return str(meta_path)
    return None


def main():
    print("=" * 60)
    print("composed 라벨 생성 + bicycle-basket-4/train 에 복사")
    print("=" * 60)

    composed_dir = Path(COMPOSED_DIR)
    train_img_dir = Path(DATASET_DIR) / "train" / "images"
    train_lbl_dir = Path(DATASET_DIR) / "train" / "labels"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list(composed_dir.glob("*_composed.jpg")) + \
             list(composed_dir.glob("*_composed.png"))
    print(f"\ncomposed 이미지 수: {len(images)}개\n")

    success, fail = 0, 0

    for img_path in sorted(images):
        # 526_m_2_composed.jpg → 526_m_2
        image_id = img_path.stem.replace("_composed", "")

        # meta 찾기
        meta_path = find_meta(image_id)
        if meta_path is None:
            print(f"  [skip] meta 없음: {image_id}")
            fail += 1
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        img_w, img_h = meta["image_size"]
        bboxes = meta["bboxes"]

        if not bboxes:
            print(f"  [warn] bbox 없음: {image_id}")

        # YOLO 라벨 생성
        lines = []
        for bb in bboxes:
            x1, y1, x2, y2 = bb["xyxy"]
            cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 이미지 복사
        dst_img = train_img_dir / img_path.name
        shutil.copy2(str(img_path), str(dst_img))

        # 라벨 저장
        dst_lbl = train_lbl_dir / (img_path.stem + ".txt")
        with open(str(dst_lbl), "w") as lf:
            lf.write("\n".join(lines))

        success += 1
        print(f"  ✅ {img_path.name}  ({len(bboxes)}개 bbox)")

    print(f"\n{'='*60}")
    print(f"완료: 성공 {success}개 / 실패 {fail}개")
    print(f"복사 위치: {train_img_dir}")


if __name__ == "__main__":
    main()