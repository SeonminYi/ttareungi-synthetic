#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[기능]
V5/* (배치폴더)/crops/*.png 파일을 모두 찾아서
./total_crops/ 폴더로 복사하는 스크립트.

*중복 방지: 파일명 앞에 배치 폴더명을 붙입니다. (예: batch_01/crops/img.png -> total_crops/batch_01_img.png)
"""

import os
import shutil

# ====================================================
# 경로 설정
# ====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 입력 폴더 (V5)
SOURCE_ROOT_DIR = os.path.join(BASE_DIR, "v5_batches")

# 출력 폴더 (total_crops)
DEST_DIR = os.path.join(BASE_DIR, "total_crops")

# 출력 폴더 생성
os.makedirs(DEST_DIR, exist_ok=True)

# ====================================================
# 메인 로직
# ====================================================

def collect_all_crops():
    if not os.path.isdir(SOURCE_ROOT_DIR):
        print(f"[ERROR] 소스 폴더를 찾을 수 없습니다: {SOURCE_ROOT_DIR}")
        return

    # V5 아래의 모든 폴더(배치들) 조회
    batch_dirs = sorted(
        d for d in os.listdir(SOURCE_ROOT_DIR)
        if os.path.isdir(os.path.join(SOURCE_ROOT_DIR, d))
    )

    print(f"[INFO] '{SOURCE_ROOT_DIR}' 내부의 총 {len(batch_dirs)}개 배치 폴더 탐색 시작...")

    count = 0

    for batch_name in batch_dirs:
        # 각 배치의 crops 폴더 경로: V5/{batch_name}/crops
        crops_path = os.path.join(SOURCE_ROOT_DIR, batch_name, "crops")

        # crops 폴더가 없으면 건너뜀
        if not os.path.isdir(crops_path):
            continue

        # crops 폴더 내의 png 파일 탐색
        files = [f for f in os.listdir(crops_path) if f.lower().endswith(".png")]

        if not files:
            continue

        for filename in files:
            src_file = os.path.join(crops_path, filename)
            
            # [중요] 파일명 중복 방지를 위해 '배치명_파일명' 형식으로 변경
            new_filename = f"{batch_name}_{filename}"
            dst_file = os.path.join(DEST_DIR, new_filename)

            shutil.copy2(src_file, dst_file)
            count += 1
        
        # 진행 상황 출력 (선택 사항)
        # print(f" - {batch_name}: {len(files)}장 복사 완료")

    print("-" * 50)
    print(f"✅ 작업 완료!")
    print(f"📂 저장 위치: {DEST_DIR}")
    print(f"📄 총 복사된 파일 수: {count}개")


if __name__ == "__main__":
    collect_all_crops()