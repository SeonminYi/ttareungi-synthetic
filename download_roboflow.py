"""
Roboflow 원본 데이터 다운로드 및 라벨 맵 구성 스크립트
(파이프라인 1단계 — 가장 먼저 실행)

사용법:
  pip install roboflow
  python download_roboflow.py
"""

import os
import re
import json
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
API_KEY   = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = "kim-6nfid"
PROJECTS  = [
    {"slug": "2nd_total",           "version": 1},
    {"slug": "for_not_empty-s8cmy", "version": 1},
]

DOWNLOAD_DIR = "./roboflow_downloads"
RF_MAP_JSON  = "./rf_map.json"  # roboflow_match.py가 이어서 사용
# ─────────────────────────────────────────


# ── Step 1: Roboflow 다운로드 ──
def download_projects():
    from roboflow import Roboflow
    rf = Roboflow(api_key=API_KEY)
    ws = rf.workspace(WORKSPACE)
    downloaded = {}
    for p in PROJECTS:
        slug, ver = p["slug"], p["version"]
        print(f"\n▶ 다운로드: {slug} (v{ver})")
        try:
            dest = os.path.join(DOWNLOAD_DIR, slug)
            project = ws.project(slug)
            version = project.version(ver)
            version.download("yolov8", location=dest, overwrite=False)
            downloaded[slug] = dest
            print(f"  ✅ {dest}")
        except Exception as e:
            print(f"  ❌ 실패: {e}")
    return downloaded


# ── Step 2: Roboflow 라벨 맵 구성 ──
def collect_roboflow_labels(downloaded: dict):
    rf_map = {}
    for slug, base_dir in downloaded.items():
        base_dir = Path(base_dir)
        for split in ["train", "valid", "test"]:
            img_dir   = base_dir / split / "images"
            label_dir = base_dir / split / "labels"
            if not img_dir.exists():
                continue
            for img_file in img_dir.iterdir():
                if not img_file.is_file():
                    continue
                clean = re.sub(
                    r'(_jpg|_JPG)\.rf\.[a-f0-9]+$', '',
                    img_file.stem, flags=re.IGNORECASE
                ).lower()

                label_file = label_dir / (img_file.stem + ".txt")
                bboxes = []
                if label_file.exists():
                    with open(label_file) as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                bboxes.append({
                                    "class": int(parts[0]),
                                    "cx": float(parts[1]),
                                    "cy": float(parts[2]),
                                    "w":  float(parts[3]),
                                    "h":  float(parts[4])
                                })
                rf_map[clean] = {
                    "image_path": str(img_file),
                    "label_path": str(label_file) if label_file.exists() else None,
                    "project":    slug,
                    "split":      split,
                    "bboxes":     bboxes
                }
    print(f"\nRoboflow 원본 총 {len(rf_map)}개")
    return rf_map


if __name__ == "__main__":
    print("="*60)
    print("Roboflow 원본 다운로드 + 라벨 맵 구성")
    print("="*60)

    downloaded = download_projects()
    if not downloaded:
        print("❌ 다운로드 실패")
        exit(1)

    rf_map = collect_roboflow_labels(downloaded)

    with open(RF_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(rf_map, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"💾 라벨 맵 저장: {RF_MAP_JSON}")
    print(f"{'='*60}")
    print(f"\n다음 단계: meta_cropping.py → prompt_v1~v4.py (Gemini 합성) 완료 후")
    print(f"           roboflow_match.py 실행")
