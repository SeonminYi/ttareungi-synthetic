#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
v5_batches/*/(edits 또는 edits_in)/*.png 에 대해:
파일명 기반 content_type을 읽고

- empty → ./empty/ 폴더로 복사
- not_empty → ./not_empty/ 폴더로 복사

하는 스크립트.
"""

import os
import re
import shutil
from typing import Optional

# ====================================================
# 기본 경로 설정
# ====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCHES_DIR = os.path.join(BASE_DIR, "prompt_800_version", "prompt_v2", "jeongho_non_empty")

# 출력 폴더
EMPTY_DIR = os.path.join(BASE_DIR, "v2_jeongho_empty")
NOT_EMPTY_DIR = os.path.join(BASE_DIR, "v2_jeongho_not_empty")

os.makedirs(EMPTY_DIR, exist_ok=True)
os.makedirs(NOT_EMPTY_DIR, exist_ok=True)

# ====================================================
# content_type 파싱
# ====================================================

CONTENT_TYPES = {
    "empty": "empty",
    "light_trash": "not_empty",
    "medium_trash": "not_empty",
    "personal_items": "not_empty",
}

CONTENT_TYPE_REGEX = re.compile(
    r"edit_(" + "|".join(CONTENT_TYPES.keys()) + r")_",
    re.IGNORECASE,
)

def parse_content_type_from_name(filename: str) -> Optional[str]:
    m = CONTENT_TYPE_REGEX.search(filename)
    if not m:
        return None
    return m.group(1).lower()


# ====================================================
# 메인 copy 함수
# ====================================================

def copy_images_to_label_folders(batches_dir: str):

    if not os.path.isdir(batches_dir):
        print(f"[ERROR] batches 디렉토리가 존재하지 않음: {batches_dir}")
        return

    batch_dirs = sorted(
        d for d in os.listdir(batches_dir)
        if os.path.isdir(os.path.join(batches_dir, d))
    )

    if not batch_dirs:
        print("[WARN] 배치 폴더 없음.")
        return

    print(f"[INFO] 총 {len(batch_dirs)}개 batch 처리 시작")

    total_count = 0

    for batch in batch_dirs:
        batch_path = os.path.join(batches_dir, batch)

        # edits 우선 → 없으면 edits_in
        edits_dir = os.path.join(batch_path, "edits")
        if not os.path.isdir(edits_dir):
            edits_dir = os.path.join(batch_path, "edits_in")

        if not os.path.isdir(edits_dir):
            continue

        files = sorted(
            f for f in os.listdir(edits_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if not files:
            continue

        print(f"\n[Batch] {batch}")

        for fname in files:
            ct_raw = parse_content_type_from_name(fname)
            if ct_raw is None:
                print(f"  [WARN] content_type 파싱 실패 → 스킵: {fname}")
                continue

            label = CONTENT_TYPES.get(ct_raw, "not_empty")

            src_path = os.path.join(edits_dir, fname)

            if label == "empty":
                dst_path = os.path.join(EMPTY_DIR, fname)
            else:
                dst_path = os.path.join(NOT_EMPTY_DIR, fname)

            shutil.copy2(src_path, dst_path)
            total_count += 1

    print(f"\n✅ 이미지 복사 완료. 총 {total_count}개 처리됨.")
    print(f" - empty 폴더:   {len(os.listdir(EMPTY_DIR))}개")
    print(f" - not_empty 폴더: {len(os.listdir(NOT_EMPTY_DIR))}개")


if __name__ == "__main__":
    copy_images_to_label_folders(BATCHES_DIR)
