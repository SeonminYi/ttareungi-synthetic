import os
import re

# ===================== 설정 =====================
MODE    = "night"  # ← "morning" 또는 "night" 으로 변경
BASE    = os.path.join(os.environ.get("TTAREUNGI_DATA_DIR", "./data"), "morning_night_test_v2")
DRY_RUN = True       # True: 목록만 출력 / False: 실제 삭제
# ================================================

# _jpg.rf.<32자리 hex>_ 패턴 (대소문자 무시: JPG.rf. 도 처리)
HASH_RE = re.compile(r'_jpe?g\.rf\.[a-f0-9]{32}_', re.IGNORECASE)


def extract_key(filename):
    """
    파일명에서 (원본ID, bbox번호) 튜플 추출.

    [test/not_empty 예시]
      630_m_2_jpg.rf.XXX_630_m_2_jpg.rf.XXX_bbox0.png
      → _jpg.rf.해시_ 가 두 번 등장 → 마지막 등장 이후: '630_m_2_jpg.rf.XXX_bbox0'
      → 다시 한 번 자르면: prefix='630_m_2', suffix='bbox0'
      → key = ('630_m_2', 'bbox0')

    [v4_non_empty 예시]
      edit_light_trash_1_594_m_3_jpg.rf.XXX_bbox0.png
      → _jpg.rf.해시_ 한 번 등장 → prefix='edit_light_trash_1_594_m_3', suffix='bbox0'
      → 숫자로 시작하는 첫 토큰부터: '594_m_3'
      → key = ('594_m_3', 'bbox0')

    [obj 형식 예시 - test/not_empty 일부]
      4089_m_not_empty_3_jpg.rf.XXX_obj2.jpg
      → prefix='4089_m_not_empty_3', suffix='obj2'
      → key = ('4089_m_not_empty_3', 'obj2')
    """
    name = os.path.splitext(filename)[0]

    # _jpg.rf.해시_ 패턴 위치를 모두 찾기
    matches = list(HASH_RE.finditer(name))
    if not matches:
        return None

    # 마지막 매치 기준으로 분리
    last_match = matches[-1]
    prefix_full = name[:last_match.start()]   # 해시 앞 전체
    suffix      = name[last_match.end():]     # 해시 뒤 (bbox0, obj2 등)

    if not suffix:
        return None

    # prefix_full 에서 원본 ID 추출
    # - test 계열: 반복 구조이므로 prefix_full 자체에도 _jpg.rf.해시_ 가 있을 수 있음
    #   → 그 뒤 부분이 실제 원본 ID
    # - v4 계열: prefix_full = 'edit_light_trash_1_594_m_3' 등
    inner = list(HASH_RE.finditer(prefix_full))
    if inner:
        # test 계열: 마지막 해시 이후 부분이 반복된 원본 ID
        origin_id = prefix_full[inner[-1].end():]
    else:
        # v4 계열: 접두사에서 숫자 시작 토큰부터 추출
        tokens = prefix_full.split('_')
        start_idx = None
        for i, tok in enumerate(tokens):
            if tok.isdigit() or (tok and tok[0].isdigit()):
                start_idx = i
                break
        if start_idx is None:
            return None
        origin_id = '_'.join(tokens[start_idx:])

    return (origin_id, suffix)


def main():
    test_dir  = os.path.join(BASE, MODE, "test",  "not_empty")
    train_dir = os.path.join(BASE, MODE, "train", "v4_non_empty")

    if not os.path.isdir(test_dir):
        print(f"[오류] 경로 없음: {test_dir}")
        return
    if not os.path.isdir(train_dir):
        print(f"[오류] 경로 없음: {train_dir}")
        return

    # ── test/not_empty 에서 키 수집 ──────────────────────────
    test_keys = set()
    skipped_test = []
    for f in os.listdir(test_dir):
        key = extract_key(f)
        if key:
            test_keys.add(key)
        else:
            skipped_test.append(f)

    print(f"[{MODE}] test/not_empty 파일 수: {len(os.listdir(test_dir))}")
    print(f"[{MODE}] 키 추출 성공: {len(test_keys)}개  /  실패: {len(skipped_test)}개")
    if skipped_test:
        print("  키 추출 실패 파일 샘플:", skipped_test[:3])
    print("  키 샘플:", sorted(test_keys)[:5])
    print()

    # ── train/v4_non_empty 순회 ──────────────────────────────
    to_delete = []
    keep      = []
    skipped_v4 = []

    for f in os.listdir(train_dir):
        key = extract_key(f)
        if key is None:
            skipped_v4.append(f)
            continue
        if key in test_keys:
            to_delete.append(f)
        else:
            keep.append(f)

    print(f"[{MODE}] v4_non_empty 파일 수: {len(os.listdir(train_dir))}")
    print(f"  삭제 대상: {len(to_delete)}개")
    print(f"  유지:     {len(keep)}개")
    print(f"  키 파싱 실패: {len(skipped_v4)}개")
    if skipped_v4:
        print("  파싱 실패 샘플:", skipped_v4[:3])
    print()

    if to_delete:
        print("── 삭제 대상 목록 ──")
        for f in sorted(to_delete):
            print(f"  {f}")
        print()

    if DRY_RUN:
        print(f"※ DRY_RUN=True → 실제 삭제하지 않음. DRY_RUN=False 로 바꾸면 삭제됩니다.")
    else:
        for f in to_delete:
            os.remove(os.path.join(train_dir, f))
        print(f"삭제 완료: {len(to_delete)}개")


if __name__ == "__main__":
    main()