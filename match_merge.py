# ===================================================================
#
#  파일: match_merge.py (Poisson 버전 / prefix 기반 원본 매칭)
#  수정: 대소문자 무시 + 이미 완료된 배치 스킵
#
# ===================================================================

import os
import json
import glob
import shutil
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# ============== 경로 설정 ==============

ORIGINAL_IMAGES_DIR = os.path.join(os.environ.get("TTAREUNGI_DATA_DIR", "./data"), "yolo_retraining", "bicycle-basket-4", "train", "images")
BATCHES_ROOT_DIR    = os.path.join(os.environ.get("TTAREUNGI_DATA_DIR", "./data"), "yolo_retraining", "v5_batches")
FINAL_OUT_DIR       = os.path.join(os.environ.get("TTAREUNGI_DATA_DIR", "./data"), "yolo_retraining", "composed_output_v2")

# ============== 매칭 파라미터 ==============

MAX_HAMMING    = 30
MIN_SSIM       = 0.30
USE_SIZE_FILTER = True
SIZE_TOL       = 0.12

# ===================================================================
#  HELPER FUNCTIONS
# ===================================================================

def safe_imread(path, flags=cv2.IMREAD_UNCHANGED):
    if not os.path.exists(path):
        return None
    try:
        n = np.fromfile(path, np.uint8)
        return cv2.imdecode(n, flags)
    except Exception as e:
        print(f"[warn] safe_imread failed for {path}: {e}")
        return None


def safe_imwrite(path, img, ext=".jpg"):
    try:
        is_success, im_buf_arr = cv2.imencode(ext, img)
        if is_success:
            im_buf_arr.tofile(path)
            return True
        return False
    except Exception as e:
        print(f"[warn] safe_imwrite failed for {path}: {e}")
        return False


def phash_of(path: str) -> Optional[int]:
    try:
        img = Image.open(path).convert("RGB")
        h_obj = imagehash.phash(img)
        return int(str(h_obj), 16)
    except Exception as e:
        print(f"[ERROR phash_of] {os.path.basename(path)}: {e}")
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def read_gray_resized(path: str, size: Tuple[int, int]) -> Optional[np.ndarray]:
    img = safe_imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def calc_priority(x1, y1, x2, y2, H, W, w_size=0.6, w_depth=0.4):
    area_norm  = ((x2 - x1) * (y2 - y1)) / max(1.0, W * H)
    depth_score = y2 / max(1.0, H)
    return w_size * area_norm + w_depth * depth_score


def ensure_mask_from_edit(edit_img, roi_w, roi_h):
    if edit_img is None:
        return None
    if edit_img.ndim == 3 and edit_img.shape[2] == 4:
        alpha = edit_img[:, :, 3]
        m = (alpha > 0).astype(np.uint8) * 255
    else:
        m = np.full((roi_h, roi_w), 255, np.uint8)
    if (m.shape[1], m.shape[0]) != (roi_w, roi_h):
        m = cv2.resize(m, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    return m

# ===================================================================
#  블렌딩 함수
# ===================================================================

def _blend_roi_alpha_gaussian_fallback(roi, edit_rgb, mask):
    mask_blur = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask_blur = cv2.GaussianBlur(mask_blur, (3, 3), 0)
    alpha = (mask_blur.astype(np.float32) / 255.0)
    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    if edit_rgb.ndim == 2:
        edit_rgb = cv2.cvtColor(edit_rgb, cv2.COLOR_GRAY2BGR)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    return (edit_rgb.astype(np.float32) * alpha +
            roi.astype(np.float32) * (1 - alpha)).astype(roi.dtype)


def blend_roi(roi, edit_rgb, mask):
    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    if edit_rgb.ndim == 2:
        edit_rgb = cv2.cvtColor(edit_rgb, cv2.COLOR_GRAY2BGR)
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        center = (cx, cy)
    else:
        center = (roi.shape[1] // 2, roi.shape[0] // 2)
    try:
        return cv2.seamlessClone(edit_rgb, roi, mask, center, cv2.NORMAL_CLONE)
    except Exception as e:
        print(f"[warn] seamlessClone failed: {e} → fallback to alpha-gaussian")
        return _blend_roi_alpha_gaussian_fallback(roi, edit_rgb, mask)

# ===================================================================
#  STEP 2a: Manifest 생성
# ===================================================================

def build_manifest(crops_dir: str, manifest_path: str) -> int:
    manifest: Dict[str, Dict] = {}
    crop_paths = sorted(glob.glob(os.path.join(crops_dir, "*.png")))
    if not crop_paths:
        print(f"[warn] build_manifest: 'crops'에 .png 없음: {crops_dir}")
    for cp in tqdm(crop_paths, desc="[Manifest]"):
        name = os.path.basename(cp)
        if "_bbox" not in name:
            continue
        stem, _ = os.path.splitext(name)
        try:
            image_id, tail = stem.split("_bbox")
            bbox_id = int(tail)
        except ValueError:
            continue
        im = safe_imread(cp, cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]
        h = phash_of(cp)
        if h is None:
            continue
        key = f"{image_id}_bbox{bbox_id}"
        manifest[key] = {
            "image_id": image_id,
            "bbox_id":  bbox_id,
            "crop_path": cp,
            "phash":    h,
            "size":     [W, H],
        }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  Manifest saved: {manifest_path}, items={len(manifest)}")
    return len(manifest)

# ===================================================================
#  STEP 2b: edits_in → edits_out 매칭
# ===================================================================

def auto_match_and_rename(edits_in, edits_out, manifest_path,
                          max_hamming, min_ssim, use_size_filter, size_tol) -> int:
    os.makedirs(edits_out, exist_ok=True)
    if not os.path.exists(manifest_path):
        return 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest:
        return 0
    entries = [{"key": k, **v, "size": tuple(v["size"])} for k, v in manifest.items()]
    used = set()
    edit_paths = sorted(glob.glob(os.path.join(edits_in, "*.png")))
    if not edit_paths:
        return 0
    matched_count = 0
    for ep in tqdm(edit_paths, desc="[Match]   "):
        eh = phash_of(ep)
        if eh is None:
            continue
        cand = [(hamming(eh, r["phash"]), r) for r in entries
                if hamming(eh, r["phash"]) <= max_hamming]
        if not cand:
            continue
        e_img = safe_imread(ep, cv2.IMREAD_COLOR)
        if e_img is None:
            continue
        eH, eW = e_img.shape[:2]
        filtered = cand
        if use_size_filter:
            filtered = [(d, r) for (d, r) in cand
                        if abs((eW / max(r["size"][0], 1)) - 1) <= size_tol
                        and abs((eH / max(r["size"][1], 1)) - 1) <= size_tol]
        if not filtered:
            filtered = cand
        best, best_score = None, -1.0
        for d, r in filtered:
            g1 = read_gray_resized(ep, r["size"])
            g2 = read_gray_resized(r["crop_path"], r["size"])
            if g1 is None or g2 is None:
                continue
            win = min(g1.shape[0], g1.shape[1], 7)
            if g1.shape[0] < win or g1.shape[1] < win or win % 2 == 0:
                continue
            score = ssim(g1, g2, data_range=g1.max() - g1.min(), win_size=win)
            if score > best_score:
                best, best_score = r, score
        if best is None or best_score < min_ssim:
            continue
        key = best["key"]
        if key in used:
            continue
        used.add(key)
        out_name = f"{best['image_id']}_bbox{best['bbox_id']}_edit.png"
        shutil.copy2(ep, os.path.join(edits_out, out_name))
        matched_count += 1
    print(f"  Match complete. Matched {matched_count} images.")
    return matched_count

# ===================================================================
#  STEP 3: 원본 이미지에 합성
# ===================================================================

def merge_one_image(image_path, meta_path, edits_dir, out_dir) -> bool:
    img = safe_imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    H, W = img.shape[:2]
    if not os.path.exists(meta_path):
        return False
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    composed = img.copy()
    occupied = np.zeros((H, W), np.uint8)
    scored = []
    for bb in meta["bboxes"]:
        x1, y1, x2, y2 = bb["xyxy"]
        scored.append((calc_priority(x1, y1, x2, y2, H, W), bb))
    scored.sort(key=lambda x: x[0], reverse=True)
    stem = meta["image_id"]
    edits_found = 0
    for _, bb in scored:
        bid  = bb["bbox_id"]
        x1, y1, x2, y2 = bb["xyxy"]
        rw, rh = x2 - x1, y2 - y1
        if rw <= 0 or rh <= 0:
            continue
        edit_path = os.path.join(edits_dir, f"{stem}_bbox{bid}_edit.png")
        mask_path = os.path.join(edits_dir, f"{stem}_bbox{bid}_mask.png")
        edit = safe_imread(edit_path, cv2.IMREAD_UNCHANGED)
        if edit is None:
            continue
        edits_found += 1
        if edit.ndim == 3 and edit.shape[2] == 4:
            edit_rgb = edit[:, :, :3]
        elif edit.ndim == 2:
            edit_rgb = cv2.cvtColor(edit, cv2.COLOR_GRAY2BGR)
        else:
            edit_rgb = edit
        if (edit_rgb.shape[1], edit_rgb.shape[0]) != (rw, rh):
            edit_rgb = cv2.resize(edit_rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
        mask = safe_imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = ensure_mask_from_edit(edit, rw, rh)
        else:
            if (mask.shape[1], mask.shape[0]) != (rw, rh):
                mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
        if mask is None:
            continue
        occ_roi = occupied[y1:y2, x1:x2]
        mask[occ_roi > 0] = 0
        if mask.sum() == 0:
            continue
        roi = composed[y1:y2, x1:x2]
        composed[y1:y2, x1:x2] = blend_roi(roi, edit_rgb, mask)
        occ_roi[mask > 0] = 255
    if edits_found == 0:
        return False
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}_composed.jpg")
    return safe_imwrite(out_path, composed, ".jpg")

# ===================================================================
#  원본 이미지 찾기 (대소문자 무시)
# ===================================================================

def find_original_image(batch_name: str, original_dir: str) -> Optional[str]:
    base_prefix = batch_name.split(".rf")[0].lower()
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    try:
        for fname in os.listdir(original_dir):
            if fname.lower().startswith(base_prefix):
                for ext in exts:
                    if fname.endswith(ext):
                        return os.path.join(original_dir, fname)
    except FileNotFoundError:
        print(f"[fail] ORIGINAL_IMAGES_DIR not found: {original_dir}")
    return None

# ===================================================================
#  전체 파이프라인 (이미 완료된 배치 스킵)
# ===================================================================

def process_all_batches(batches_root_dir: str, final_out_dir: str):
    print("======== 🚀 STARTING BATCH PROCESSING (Match & Merge - Poisson) ========")
    os.makedirs(final_out_dir, exist_ok=True)

    batch_folders = sorted(glob.glob(os.path.join(batches_root_dir, "*")))
    batch_folders = [b for b in batch_folders if os.path.isdir(b)]

    if not batch_folders:
        print(f"⚠️ WARNING: No batch folders found in {batches_root_dir}.")
        return

    print(f"Found {len(batch_folders)} batches to process...")
    total_ok, total_fail, total_skip = 0, 0, 0

    for batch_dir in batch_folders:
        batch_name = os.path.basename(batch_dir)

        # ✅ 이미 완료된 배치 스킵
        meta_path_check = os.path.join(batch_dir, "meta", f"{batch_name}.json")
        if os.path.exists(meta_path_check):
            with open(meta_path_check, "r", encoding="utf-8") as f:
                image_id = json.load(f).get("image_id", "")
            out_path = os.path.join(final_out_dir, f"{image_id}_composed.jpg")
            if os.path.exists(out_path):
                print(f"[skip] 이미 완료: {image_id}")
                total_skip += 1
                total_ok += 1
                continue

        print(f"\n--- Processing Batch: {batch_name} ---")

        crops_dir     = os.path.join(batch_dir, "crops")
        edits_in_dir  = os.path.join(batch_dir, "edits_in")
        meta_dir      = os.path.join(batch_dir, "meta")
        manifest_path = os.path.join(batch_dir, "manifest.json")
        edits_out_dir = os.path.join(batch_dir, "edits_out")

        if not os.path.exists(crops_dir):
            print("[skip] 'crops' 폴더 없음.")
            total_fail += 1
            continue

        item_count = build_manifest(crops_dir, manifest_path)
        if item_count == 0:
            print("[skip] manifest item 0개.")
            total_fail += 1
            continue

        if not os.path.exists(edits_in_dir):
            print("[skip] 'edits_in' 폴더 없음.")
            total_fail += 1
            continue

        matched_count = auto_match_and_rename(
            edits_in_dir, edits_out_dir, manifest_path,
            MAX_HAMMING, MIN_SSIM, USE_SIZE_FILTER, SIZE_TOL)

        if matched_count == 0:
            print("[info] 매칭된 edit 없음. 스킵.")
            total_ok += 1
            continue

        meta_path = os.path.join(meta_dir, f"{batch_name}.json")
        if not os.path.exists(meta_path):
            print(f"[skip] Meta file not found: {meta_path}")
            total_fail += 1
            continue

        original_img_path = find_original_image(batch_name, ORIGINAL_IMAGES_DIR)
        if original_img_path is None:
            print(f"[skip] Original image not found: {batch_name}")
            total_fail += 1
            continue

        try:
            ok = merge_one_image(original_img_path, meta_path, edits_out_dir, final_out_dir)
        except Exception as e:
            print(f"[ERROR] {e}")
            ok = False

        if ok:
            print(f"✅ {batch_name} merged successfully.")
            total_ok += 1
        else:
            print(f"❌ {batch_name} merge failed.")
            total_fail += 1

    print("\n======== ✅ BATCH PROCESSING FINISHED (Poisson) ========")
    print(f"Success: {total_ok} (스킵 포함: {total_skip}), Fail: {total_fail}")
    print(f"Final composed images are in: {final_out_dir}")


if __name__ == "__main__":
    process_all_batches(BATCHES_ROOT_DIR, FINAL_OUT_DIR)