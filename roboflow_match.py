"""
Roboflow 다운로드 + 데이터 매칭 + 구조 분석 스크립트 v3
(패턴3 추가: 1708_m_not_empty_1_jpg.rf.xxx_obj3 형태 지원)

사용법:
  pip install roboflow
  python roboflow_match_v3.py
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict, Counter

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
API_KEY   = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = "kim-6nfid"
PROJECTS  = [
    {"slug": "2nd_total",           "version": 1},
    {"slug": "for_not_empty-s8cmy", "version": 1},
]

SYNTHETIC_DIR = os.path.join(os.environ.get("TTAREUNGI_DATA_DIR", "./data"), "final_training_version", "final_training_v4", "v4_non_empty")
ONLY_REAL_DIR = os.path.join(os.environ.get("TTAREUNGI_DATA_DIR", "./data"), "only_real")

DOWNLOAD_DIR = "./roboflow_downloads"
RESULT_JSON  = "./match_result_v3.json"
# ─────────────────────────────────────────


def extract_base_name(filename: str):
    """
    파일명 → 원본 이미지 베이스명 (526_m_2 형태)

    패턴1: 526_m_2_jpg.rf.xxx_526_m_2_jpg.rf.xxx_bbox1.png
    패턴2: edit_xxx_526_m_2_jpg.rf.xxx_bbox1.png
    패턴3: 1708_m_not_empty_1_jpg.rf.xxx_obj3.jpg  (only_real not_empty)
    """
    stem = Path(filename).stem

    # 패턴1: 숫자시작 반복 구조
    m = re.match(
        r'^(\d+_[a-zA-Z]+_\d+)_[a-zA-Z]+\.rf\.[a-f0-9]+_'
        r'\1_[a-zA-Z]+\.rf\.[a-f0-9]+_bbox\d+$',
        stem, re.IGNORECASE
    )
    if m:
        return m.group(1).lower()

    # 패턴2: edit_xxx_ prefix
    m = re.search(
        r'(\d+_[a-zA-Z]+_\d+)_[a-zA-Z]+\.rf\.[a-f0-9]+_bbox\d+$',
        stem, re.IGNORECASE
    )
    if m:
        return m.group(1).lower()

    # 패턴3: 1708_m_not_empty_1_jpg.rf.xxx_obj3
    # _not_empty_ or _empty_ 제거 후 숫자_알파벳_숫자 추출
    m = re.match(
        r'^(\d+_[a-zA-Z]+)(?:_not_empty|_empty)?_(\d+)_[a-zA-Z]+\.rf\.[a-f0-9]+_obj\d+$',
        stem, re.IGNORECASE
    )
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"

    return None


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


# ── Step 2: Roboflow 라벨 맵 ──
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


# ── Step 3: 합성 데이터 매칭 ──
def match_synthetic(rf_map: dict):
    synth_dir = Path(SYNTHETIC_DIR)
    if not synth_dir.exists():
        print(f"  [경고] 합성 폴더 없음: {synth_dir}")
        return {}, set()

    matched    = {}
    unmatched  = []
    used_bases = set()

    for f in synth_dir.iterdir():
        if not f.is_file():
            continue
        base = extract_base_name(f.name)
        if base and base in rf_map:
            matched[f.name] = {
                "base_name":  base,
                "project":    rf_map[base]["project"],
                "image_path": rf_map[base]["image_path"],
                "label_path": rf_map[base]["label_path"],
                "bboxes":     rf_map[base]["bboxes"],
            }
            used_bases.add(base)
        else:
            unmatched.append({"filename": f.name, "base_name": base})

    total = len(matched) + len(unmatched)
    print(f"\n[합성 데이터 매칭]")
    print(f"  전체  : {total}개")
    print(f"  성공  : {len(matched)}개  ({len(matched)/total*100:.1f}%)")
    print(f"  실패  : {len(unmatched)}개  ({len(unmatched)/total*100:.1f}%)")
    print(f"  사용된 원본 수: {len(used_bases)}개")
    if unmatched:
        print(f"  실패 샘플:")
        for u in unmatched[:5]:
            print(f"    - {u['filename']} → {u['base_name']}")
    return matched, used_bases


# ── Step 4: only_real 매칭 ──
def match_only_real(rf_map: dict):
    only_real = Path(ONLY_REAL_DIR)
    result = {"train": {}, "test": {}}

    for split in ["train", "test"]:
        for label in ["empty", "not_empty"]:
            folder = only_real / split / label
            if not folder.exists():
                print(f"  [경고] 폴더 없음: {folder}")
                continue

            matched_files   = []
            unmatched_files = []

            for f in folder.iterdir():
                if not f.is_file():
                    continue
                base = extract_base_name(f.name)
                if base and base in rf_map:
                    matched_files.append({
                        "filename":   f.name,
                        "base_name":  base,
                        "image_path": str(f),
                        "rf_image":   rf_map[base]["image_path"],
                        "label_path": rf_map[base]["label_path"],
                        "bboxes":     rf_map[base]["bboxes"],
                        "project":    rf_map[base]["project"],
                    })
                else:
                    unmatched_files.append({
                        "filename":  f.name,
                        "base_name": base
                    })

            total = len(matched_files) + len(unmatched_files)
            result[split][label] = {
                "total":           total,
                "matched":         len(matched_files),
                "unmatched":       len(unmatched_files),
                "files":           matched_files,
                "unmatched_files": unmatched_files
            }

    print(f"\n[only_real 매칭]")
    for split in ["train", "test"]:
        for label in ["empty", "not_empty"]:
            d = result[split].get(label, {})
            if d.get("total", 0) > 0:
                print(f"  {split}/{label}: {d['total']}개  "
                      f"(매칭 {d['matched']} / 실패 {d['unmatched']})")
            if d.get("unmatched_files"):
                for u in d["unmatched_files"][:3]:
                    print(f"    실패 샘플: {u['filename']} → {u['base_name']}")
    return result


# ── Step 5: 테스트 후보 분석 ──
def analyze_test_candidates(rf_map: dict, used_bases: set):
    unused = {k: v for k, v in rf_map.items() if k not in used_bases}

    empty_cnt     = sum(1 for v in unused.values() if len(v["bboxes"]) == 0)
    not_empty_cnt = sum(1 for v in unused.values() if len(v["bboxes"]) > 0)

    print(f"\n[테스트셋 후보 분석]")
    print(f"  Roboflow 전체   : {len(rf_map)}개")
    print(f"  합성에 쓰인 것  : {len(used_bases)}개")
    print(f"  테스트 후보     : {len(unused)}개")
    print(f"    └ empty       : {empty_cnt}개")
    print(f"    └ not_empty   : {not_empty_cnt}개")

    proj_cnt = Counter(v["project"] for v in unused.values())
    for proj, cnt in proj_cnt.items():
        print(f"    [{proj}] {cnt}개")

    return unused


# ── 메인 ──
if __name__ == "__main__":
    print("="*60)
    print("Roboflow 다운로드 + 데이터 구조 분석 v3")
    print("="*60)

    downloaded = download_projects()
    if not downloaded:
        print("❌ 다운로드 실패")
        exit(1)

    rf_map          = collect_roboflow_labels(downloaded)
    synth_matched, used_bases = match_synthetic(rf_map)
    real_matched    = match_only_real(rf_map)
    test_candidates = analyze_test_candidates(rf_map, used_bases)

    result = {
        "summary": {
            "roboflow_total":       len(rf_map),
            "synthetic_matched":    len(synth_matched),
            "synthetic_used_bases": len(used_bases),
            "test_candidates":      len(test_candidates),
            "only_real": {
                split: {
                    label: {
                        "total":     real_matched[split][label]["total"],
                        "matched":   real_matched[split][label]["matched"],
                        "unmatched": real_matched[split][label]["unmatched"],
                    }
                    for label in ["empty", "not_empty"]
                    if real_matched[split].get(label, {}).get("total", 0) > 0
                }
                for split in ["train", "test"]
            }
        },
        "synthetic_matched":  synth_matched,
        "only_real_matched":  real_matched,
        "test_candidates":    test_candidates,
    }

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"💾 결과 저장: {RESULT_JSON}")
    print(f"{'='*60}")
    print(f"\n다음 단계:")
    print(f"  1. only_real train/not_empty 매칭률 확인")
    print(f"  2. 테스트셋 비율 결정 (empty vs not_empty)")
    print(f"  3. 실험 A/B 데이터셋 구성 + YOLOv8n 학습")