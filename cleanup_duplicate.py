"""
composed_output/ 에서 중복 파일 정리
고유한 image_id 기준으로 1개만 남기고 나머지 삭제

파일명 예시:
  526_m_2_composed.jpg  ← 정상
  중복이 있다면 같은 image_id로 여러 파일이 있을 수 있음
"""

import os
from pathlib import Path
from collections import defaultdict

BASE = os.environ.get("TTAREUNGI_DATA_DIR", "./data")
COMPOSED_DIR = os.path.join(BASE, "composed_output")


def main():
    composed = Path(COMPOSED_DIR)
    files = list(composed.glob("*_composed.jpg")) + list(composed.glob("*_composed.png"))
    print(f"전체 파일 수: {len(files)}개")

    # image_id 기준으로 그룹화
    groups = defaultdict(list)
    for f in files:
        image_id = f.stem.replace("_composed", "")
        groups[image_id].append(f)

    print(f"고유 image_id 수: {len(groups)}개")

    # 중복 확인 및 삭제 (가장 오래된 것 1개만 남김)
    deleted = 0
    for image_id, file_list in groups.items():
        if len(file_list) > 1:
            # 수정 시간 기준 오래된 것 유지, 나머지 삭제
            file_list.sort(key=lambda f: f.stat().st_mtime)
            keep = file_list[0]
            for f in file_list[1:]:
                f.unlink()
                deleted += 1
                print(f"  삭제: {f.name}")

    print(f"\n삭제: {deleted}개")
    remaining = len(list(composed.glob("*_composed.jpg"))) + len(list(composed.glob("*_composed.png")))
    print(f"남은 파일: {remaining}개")


if __name__ == "__main__":
    main()