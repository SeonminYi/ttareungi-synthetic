"""
valid/images 파일명 기준으로
v5_batches에서 겹치는 배치 폴더 전체 삭제
"""

import os
import re
import shutil

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
VALID_IMAGES_DIR = os.path.join(os.environ.get("TTAREUNGI_DATA_DIR", "./data"), "yolo_retraining", "bicycle-basket-4", "valid", "images")
V5_BATCHES_DIR   = os.path.join(os.environ.get("TTAREUNGI_DATA_DIR", "./data"), "yolo_retraining", "v5_batches")
# ─────────────────────────────────────────


def get_base_key(filename: str) -> str:
    """
    파일명 → 핵심 키 추출
    1405_n_1_JPG.rf.xxx.jpg → 1405_n_1
    1405_n_1_jpg.rf.xxx     → 1405_n_1  (폴더명)
    """
    stem = os.path.splitext(filename)[0]
    # .rf. 앞까지만 (jpg/JPG 포함)
    m = re.match(r'^(\d+_[a-zA-Z]+_\d+)_[a-zA-Z]+\.rf\.',
                 stem, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # .rf. 없으면 첫 3파트만
    parts = stem.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}".lower()
    return stem.lower()


def main():
    print("=" * 60)
    print("v5_batches 필터링 (valid 기준)")
    print("=" * 60)

    # 1. valid 키 수집
    valid_keys = set()
    for f in os.listdir(VALID_IMAGES_DIR):
        key = get_base_key(f)
        valid_keys.add(key)
    print(f"\nvalid 키 수: {len(valid_keys)}개")
    print("샘플:", list(valid_keys)[:5])

    # 2. v5_batches 배치 폴더 확인
    batch_dirs = [
        d for d in os.listdir(V5_BATCHES_DIR)
        if os.path.isdir(os.path.join(V5_BATCHES_DIR, d))
    ]
    print(f"\nv5_batches 배치 폴더 수: {len(batch_dirs)}개")

    # 3. 겹치는 배치 폴더 찾기
    to_remove = []
    for batch in batch_dirs:
        key = get_base_key(batch)
        if key in valid_keys:
            to_remove.append(batch)

    print(f"\n겹치는 배치 폴더: {len(to_remove)}개")
    for b in to_remove:
        print(f"  - {b}")

    if not to_remove:
        print("\n겹치는 배치 없음. 종료.")
        return

    # 4. 삭제 확인
    confirm = input(f"\n{len(to_remove)}개 배치 폴더를 삭제할까요? (yes 입력): ")
    if confirm.strip().lower() == "yes":
        for batch in to_remove:
            batch_path = os.path.join(V5_BATCHES_DIR, batch)
            shutil.rmtree(batch_path)
            print(f"  🔥 삭제: {batch}")
        print(f"\n✅ {len(to_remove)}개 배치 폴더 삭제 완료")
        remaining = len(batch_dirs) - len(to_remove)
        print(f"남은 배치 폴더: {remaining}개")
    else:
        print("취소됨")


if __name__ == "__main__":
    main()