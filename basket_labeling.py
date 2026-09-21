#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
v5_batches/*/(edits 또는 edits_in)/*.png 에 대해
파일 이름의 content_type을 기반으로
- image_path
- batch_name
- content_type (empty/light_trash/medium_trash/personal_items)
- label (empty/not_empty)
를 CSV로 저장하는 스크립트.
"""

import os
import re
import csv
from typing import Optional

# ===============================
# 설정
# ===============================

# 이 스크립트가 있는 디렉토리를 기준으로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# v5_batches 디렉토리 (예: BASE_DIR/v5_batches/1501_m_1.../edits/*.png)
BATCHES_DIR = os.path.join(BASE_DIR, "empty")

# 출력 CSV 파일 경로
CSV_OUTPUT = os.path.join(BASE_DIR, "synthetic_basket_labels.csv")

# per-image label txt 만들고 싶으면 True
SAVE_PER_IMAGE_TXT = False


# ===============================
# content_type 파싱
# ===============================

# content_type → binary label 매핑
CONTENT_TYPES = {
    "empty": 0,
    "light_trash": 1,
    "medium_trash": 1,
    "personal_items": 1,
}

CONTENT_TYPE_REGEX = re.compile(
    r"edit_(" + "|".join(CONTENT_TYPES.keys()) + r")_",
    re.IGNORECASE,
)


def parse_content_type_from_name(filename: str) -> Optional[str]:
    """
    파일 이름에서 content_type을 정규식으로 파싱.
    예: edit_empty_0001.png, edit_light_trash_0002.png 등
    """
    m = CONTENT_TYPE_REGEX.search(filename)
    if not m:
        return None
    return m.group(1).lower()


# ===============================
# empty/not_empty로 변환
# ===============================

def content_type_to_string(ct: str) -> str:
    """
    content_type 문자열을 binary label("empty" / "not_empty")로 변환.
    정의되지 않은 content_type이 들어오면 기본값으로 "not_empty"를 반환한다.
    """
    return CONTENT_TYPES.get(ct.lower(), "not_empty")


# ===============================
# 메인 라벨링 함수
# ===============================

def generate_labels_for_edits(batches_dir: str, csv_output: str):
    rows = []

    # v5_batches 디렉토리 내의 batch 폴더들(예: 1501_m_1_jpg.rf...)을 순회
    if not os.path.isdir(batches_dir):
        print(f"[ERROR] batches 디렉토리가 존재하지 않습니다: {batches_dir}")
        return

    batch_names = sorted(
        d for d in os.listdir(batches_dir)
        if os.path.isdir(os.path.join(batches_dir, d))
    )

    if not batch_names:
        print(f"[WARN] batches 디렉토리 내에 하위 폴더가 없습니다: {batches_dir}")
        return

    print(f"[INFO] 총 {len(batch_names)}개 batch에서 edits 이미지를 탐색합니다.")

    for batch_name in batch_names:
        batch_path = os.path.join(batches_dir, batch_name)

        # 우선순위: edits > edits_in
        edits_dir = os.path.join(batch_path, "edits")
        subdir_name = "edits"

        if not os.path.isdir(edits_dir):
            edits_dir = os.path.join(batch_path, "edits_in")
            subdir_name = "edits_in"

        if not os.path.isdir(edits_dir):
            # edits / edits_in 둘 다 없으면 스킵
            continue

        files = sorted(
            f for f in os.listdir(edits_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not files:
            continue

        print(f"\n[Batch] {batch_name} / 사용 디렉토리: {subdir_name}")

        for fname in files:
            ct = parse_content_type_from_name(fname)
            if ct is None:
                print(f"  [WARN] content_type 파싱 실패: {fname}")
                continue

            label_str = content_type_to_string(ct)

            # CSV에 넣을 상대 경로 (v5_batches 기준)
            rel_path = os.path.join("v5_batches", batch_name, subdir_name, fname)

            # CSV용 row 추가: image_path, batch_name, content_type, label
            rows.append((rel_path, batch_name, ct, label_str))

            # per-image txt 파일 저장 옵션
            if SAVE_PER_IMAGE_TXT:
                txt_path = os.path.join(
                    edits_dir, os.path.splitext(fname)[0] + ".txt"
                )
                with open(txt_path, "w", encoding="utf-8") as lf:
                    lf.write(label_str)

    # CSV 저장
    if not rows:
        print("⚠️ 생성된 라벨이 없습니다.")
        return

    with open(csv_output, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["image_path", "batch_name", "content_type", "label"])
        writer.writerows(rows)

    print(f"\n✅ CSV 저장 완료: {csv_output}")
    print(f"총 {len(rows)}개 이미지에 대해 라벨 생성됨.")


if __name__ == "__main__":
    generate_labels_for_edits(BATCHES_DIR, CSV_OUTPUT)
