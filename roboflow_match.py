"""
합성 데이터 매칭 + 구조 분석 스크립트
(파이프라인 3→4단계 사이 — Gemini 합성이 끝난 뒤 실행)

사전 조건:
  download_roboflow.py 실행 완료 (rf_map.json 존재)
  prompt_v1~v4.py로 합성 이미지 생성 완료

사용법:
  python roboflow_match.py
"""

import os
import re
import json
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
RF_MAP_JSON   = "./rf_map.json"  # download_roboflow.py의 출력

SYNTHETIC_DIR = os.path.join(os.environ.get("TTAREUNGI_DATA_DIR", "./data"), "final_training_version", "final_training_v4", "v4_non_empty")
ONLY_REAL_DIR = os.path.join(os.environ.get("TTAREUNGI_DATA_DIR", "./data"), "only_real")

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
    m = re.match(
        r'^(\d+_[a-zA-Z]+)(?:_not_empty|_empty)?_(\d+)_[a-zA-Z]+\.rf\.[a-f0-9]+_obj\d+$',
        stem, re.IGNORECASE
    )
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"

    return None


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
    print("합성 데이터 매칭 + 구조 분석")
    print("="*60)

    if not os.path.exists(RF_MAP_JSON):
        print(f"❌ {RF_MAP_JSON} 없음. download_roboflow.py를 먼저 실행하세요.")
        exit(1)

    with open(RF_MAP_JSON, "r", encoding="utf-8") as f:
        rf_map = json.load(f)
    print(f"라벨 맵 로드 완료: {len(rf_map)}개")

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
    print(f"  3. prepare_batches.py 실행")
