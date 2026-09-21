# ===================================================================
#
#  파일 1: meta_cropping.py
#
#  (step1_crop_and_meta.py)
#
#  1. 원본 YOLOv8 데이터셋을 스캔합니다.
#  2. 원본 이미지 'ImageXXX'별로 'batches/ImageXXX' 폴더를 생성합니다.
#  3. 'batches/ImageXXX/crops/'에 크롭 이미지를 저장합니다.
#  4. 'batches/ImageXXX/meta/'에 메타데이터(.json)를 저장합니다.
#  5. 'batches/ImageXXX/edits_in/' 빈 폴더를 생성합니다. (Gemini 작업 대기)
#
# ===================================================================


from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("kim-6nfid").project("jeongho_labeling-wl3wd")
version = project.version(2)
dataset = version.download("yolov8")
                
                
                

import os, json, glob
from dataclasses import dataclass, asdict
from typing import List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# ============== 경로 설정 ==============
# 원본 YOLOv8 데이터셋 루트
DATASET_DIR = "Jeongho_labeling-2"  # ← 수정 1: 실제 다운로드된 폴더 이름
SPLITS = ["train", "valid", "test"]
# "Image_0001", "Image_0002" 등 개별 배치 폴더가 생성될 루트
BATCHES_ROOT_DIR = "batches" 

# ============== 유틸 ==============
def yolo_to_xyxy(xc, yc, w, h, W, H):
    x, y = xc*W, yc*H
    bw, bh = w*W, h*H
    x1, y1 = int(round(x - bw/2)), int(round(y - bh/2))
    x2, y2 = int(round(x + bw/2)), int(round(y + bh/2))
    x1, y1 = max(0, min(x1, W-1)), max(0, min(y1, H-1))
    x2, y2 = max(0, min(x2, W-1)), max(0, min(y2, H-1))
    return x1, y1, x2, y2

def xyxy_area(x1,y1,x2,y2):
    return max(0, x2-x1) * max(0, y2-y1)

def safe_imread(path, flags=cv2.IMREAD_UNCHANGED):
    if not os.path.exists(path): return None
    try:
        n = np.fromfile(path, np.uint8)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(f"[warn] safe_imread failed for {path}: {e}")
        return None

def safe_imwrite(path, img):
    try:
        is_success, im_buf_arr = cv2.imencode(".png", img)
        if is_success:
            im_buf_arr.tofile(path)
            return True
        return False
    except Exception as e:
        print(f"[warn] safe_imwrite failed for {path}: {e}")
        return False

# ============== 데이터 클래스 ==============
@dataclass
class BBoxMeta:
    bbox_id: int
    cls: int
    xyxy: Tuple[int,int,int,int]
    area: int
    bottom_y: int

@dataclass
class ImageMeta:
    image_id: str
    width: int
    height: int
    bboxes: List[BBoxMeta]

# ============== Step1: YOLOv8 → 배치별 크롭 + 메타 저장 ==============
def create_batch_structure(dataset_dir, splits, batches_root_dir):
    """
    Step 1: 원본 YOLO 데이터셋을 읽어, 원본 이미지별로
    'batches/ImageXXX/' 폴더 구조(crops, meta, edits_in)를 생성합니다.
    """
    os.makedirs(batches_root_dir, exist_ok=True)
    print(f"[Step 1]\n  Input Dataset: {dataset_dir}\n  Output Batches: {batches_root_dir}")

    total_images = 0
    total_bboxes = 0

    for sp in splits:
        img_dir = os.path.join(dataset_dir, sp, "images")
        lbl_dir = os.path.join(dataset_dir, sp, "labels")
        if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
            print(f"[skip] split '{sp}' 경로 없음:", img_dir, lbl_dir); continue

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        for img_path in tqdm(img_paths, desc=f"[Step 1 Crop] {sp}"):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            
            # --- 1. 원본 이미지별로 개별 배치 폴더 생성 ---
            current_batch_dir = os.path.join(batches_root_dir, stem)
            crops_out_dir = os.path.join(current_batch_dir, "crops")
            edits_in_dir = os.path.join(current_batch_dir, "edits")
            meta_out_dir = os.path.join(current_batch_dir, "meta")
            
            os.makedirs(crops_out_dir, exist_ok=True)
            os.makedirs(edits_in_dir, exist_ok=True) # Gemini 작업용 빈 폴더
            os.makedirs(meta_out_dir, exist_ok=True)
            # --- ---

            label_path = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(label_path):
                continue

            img = safe_imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print("[warn] read fail:", img_path); continue
            H, W = img.shape[:2]
            total_images += 1

            with open(label_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            bboxes = []
            for i, ln in enumerate(lines):
                parts = ln.split()
                if len(parts) < 5: continue
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                x1,y1,x2,y2 = yolo_to_xyxy(xc,yc,w,h,W,H)
                area = xyxy_area(x1,y1,x2,y2)
                if area <= 0: 
                    continue

                crop = img[y1:y2, x1:x2]
                crop_name = f"{stem}_bbox{i}.png"
                
                # 2. 'crops' 폴더에 크롭 저장
                safe_imwrite(os.path.join(crops_out_dir, crop_name), crop)

                bboxes.append(BBoxMeta(
                    bbox_id=i, cls=cls, xyxy=(x1,y1,x2,y2), area=area, bottom_y=y2
                ))
                total_bboxes += 1

            if bboxes:
                meta = ImageMeta(stem, W, H, bboxes)
                # 3. 'meta' 폴더에 메타데이터(json) 저장
                meta_path = os.path.join(meta_out_dir, f"{stem}.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "image_id": meta.image_id,
                        "width": meta.width,
                        "height": meta.height,
                        "bboxes": [asdict(bb) for bb in meta.bboxes]
                    }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Step 1 완료: {total_images}개 이미지에서 {total_bboxes}개 크롭 생성 완료.")
    print(f"  배치 폴더: {batches_root_dir}")


if __name__ == "__main__":
    create_batch_structure(DATASET_DIR, SPLITS, BATCHES_ROOT_DIR)