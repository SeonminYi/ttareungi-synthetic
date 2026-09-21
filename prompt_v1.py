#!/usr/bin/env python
# -*- coding: utf-8 -*-
# V15

"""
prompt.py  (google-genai 1.51.0 기준)

사용:
    (capstone 폴더에서)
    python prompt.py

동작:
    - ./batches/ 아래의 모든 배치 폴더를 순회
    - 각 배치 폴더 안의 crops/*.png 에 대해
        1) content_type / intensity / damage 샘플링
        2) (empty 제외) 프롬프트 생성
        3) (empty 제외) gemini-2.5-flash-image 에 이미지 편집 요청
        4) 결과 이미지를 같은 배치의 edits/ 에 저장
           파일명: edit_{content_type}_{intensity}_{damage}_{원본파일명}

    - content_type 이 empty 인 경우:
        → Gemini 호출 없이 원본 이미지를 그대로 edits/ 에 복사 저장
"""

import os
import random
from io import BytesIO
from collections import Counter

import numpy as np
from PIL import Image

from google import genai
from google.genai import errors


# ===============================
# 1. Gemini 클라이언트
# ===============================

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

MODEL_NAME = "gemini-3-pro-image-preview"


# ===============================
# 2. 디렉토리 설정
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCHES_DIR = os.path.join(BASE_DIR, "batches")


# ===============================
# 3. content_type / intensity 설정
# ===============================

CONFIG = {
    "empty": {
        "prob": 0.1,
        "min": 0,
        "max": 0,
        "mu": 0.0,
        "sigma": 0.0,
    },
    "light_trash": {
        "prob": 0.3,
        "min": 1,
        "max": 4,
        "mu": 2.0,
        "sigma": 1.0,
    },
    "medium_trash": {
        "prob": 0.3,  
        "min": 2,
        "max": 5,
        "mu": 3.0,
        "sigma": 1.5,
    },
    "personal_items": {
        "prob": 0.3,
        "min": 1,
        "max": 3,   
        "mu": 2.0,
        "sigma": 0.7,
    }
}


def sample_content_type() -> str:
    names = list(CONFIG.keys())
    probs = [CONFIG[name]["prob"] for name in names]
    return random.choices(names, probs, k=1)[0]


def sample_intensity(content_type: str) -> int:
    cfg = CONFIG[content_type]
    if cfg["min"] == cfg["max"]:
        return cfg["min"]
    x = np.random.normal(loc=cfg["mu"], scale=cfg["sigma"])
    x = max(cfg["min"], min(cfg["max"], x))
    return int(round(x))


# intensity → 말로 풀어준 표현 (프롬프트용)
def intensity_to_phrase(intensity: int) -> str:
    if intensity <= 0:
        return "no items"
    if intensity == 1:
        return "about one item"
    if intensity == 2:
        return "about two items"
    if intensity <= 4:
        return f"around {intensity} small items"
    return f"several items (around {intensity})"


# ===============================
# 4. damage 설정 (light/medium에만 적용)
# ===============================

DAMAGE_CONFIG = {
    "clean": 0.4,
    "slightly_dirty": 0.2,
    "stained": 0.2,
    "crumpled": 0.2
}

DAMAGE_PHRASES = {
    "clean": "The items look fairly clean, recently used but not strongly damaged.",
    "slightly_dirty": "The items look slightly dirty or used, with a few small stains or smudges.",
    "stained": "Some items are noticeably stained or splattered, like they were used for food or drinks.",
    "crumpled": "Several wrappers or cups look crumpled and squashed as if they were thrown away carelessly."
}


def sample_damage(content_type: str) -> str:
    # empty, personal_items 에는 damage 적용하지 않음
    if content_type in ("empty", "personal_items"):
        return "none"
    names = list(DAMAGE_CONFIG.keys())
    probs = [DAMAGE_CONFIG[n] for n in names]
    return random.choices(names, probs, k=1)[0]


# ===============================
# 5. object 리스트 로딩 (txt)
#    - txt에는 색/손상 포함 X
#    - 한 줄당 "plastic bottle" 이런 식의 기본 객체 이름만
# ===============================

OBJECT_FILES = {
    "light_trash": "objects_light_trash.txt",
    "medium_trash": "objects_medium_trash.txt",
    "personal_items": "objects_personal_items.txt",
}


def load_object_lists() -> dict[str, list[str]]:
    obj_dict: dict[str, list[str]] = {}
    for ctype, fname in OBJECT_FILES.items():
        path = os.path.join(BASE_DIR, fname)
        items: list[str] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        items.append(name)
        obj_dict[ctype] = items
    return obj_dict


OBJECT_ITEMS = load_object_lists()


def sample_object_examples(content_type: str, intensity: int) -> str:
    """
    txt에서 불러온 object 목록 중 몇 개를 샘플해서
    'Use realistic objects such as ...' 문장에 쓸 문자열 생성.

    txt에는 색상/손상 같은 디테일은 넣지 않고,
    'plastic bottle', 'paper cup' 같은 베이스 이름만 넣는 전제.
    """
    items = OBJECT_ITEMS.get(content_type, [])
    if not items:
        return ""

    # intensity 기반으로 1~(intensity+1)개 정도 샘플 (최소 1개)
    k = max(1, min(intensity + 1, len(items)))
    chosen = random.sample(items, k=k)

    if len(chosen) == 1:
        return chosen[0]
    else:
        # 영어 문장 연결: a, b, and c
        return ", ".join(chosen[:-1]) + ", and " + chosen[-1]


# ===============================
# 6. content_type 별 기본 설명
#    - 여기서는 "행동 지시"만, 예시는 txt에서.
# ===============================

CONTENT_PHRASES = {
    "empty": (
        "Do not add any new objects. Leave the inside of the basket completely empty."
    ),
    "light_trash": (
        "Add a small number of light trash items inside the basket."
    ),
    "medium_trash": (
        "Fill part of the basket with mixed trash items inside the basket, but do not overflow it."
    ),
    "personal_items": (
        "Place a few personal belongings inside the basket. Their surfaces should look mostly matte "
        "or softly reflective, avoiding very strong glossy highlights."
    )
}


# ===============================
# 7. 프롬프트 템플릿
# ===============================

PROMPT_TEMPLATE = """
You are editing a cropped photo of a public rental bicycle basket.
Use the input photo as the base and only add or remove small objects inside the basket.

Goals:
- Keep all original pixels of the metal basket wires, screws, bike frame, wheel, and background unchanged.
- Any new object must appear physically inside the basket, **RESTING ON THE BASKET FLOOR OR STACKED REALISTICALLY ON TOP OF OTHER OBJECTS.** **DO NOT place objects below the visible basket floor or wires, or allow them to be unnaturally suspended.**
- **CRITICAL: New objects MUST NOT be wedged or stuck unnaturally between the wire grids or the basket frame. Ensure stable placement.**
- **ABSOLUTE RULE: The entire silhouette of any added object MUST be fully contained within the visible physical boundaries and silhouette of the basket.** **DO NOT extend any part of the object below the lowest visible wire or outside the side perimeter.**
- **CRITICAL REQUIREMENT:** **OBJECTS MUST BE OCLLUDED BY THE BASKET WIRES.** If any new object overlaps the metal grid, **THE METAL WIRES MUST ALWAYS BE VISIBLE IN FRONT OF THE OBJECT** (the object is occluded by the grid, never painting over the wires). Ensure the metal lines precisely cover the added object, as if the object is genuinely resting deep inside the basket.
- **CRITICAL: MATCH THE EXACT IMAGE FIDELITY AND QUALITY OF THE ORIGINAL PHOTO.** New objects **MUST NOT look sharper, cleaner, or higher resolution** than the existing scene.
- If you need to regenerate or refine the visibility of the metal grid over the new objects due to low resolution, ensure the regenerated wires exactly match the original metal's thickness, color, texture, and blur level. The added grid must maintain the low fidelity of the original photo.

Edit instructions:
- {content_sentence}
- The total amount of items should match this level: {intensity_phrase}.
- {damage_sentence}

Object appearance:
- Use realistic everyday objects, with varied but natural colors such as muted reds, greens, blues, and grays.
- Colors of different objects should not all be identical; allow subtle color diversity.
- Avoid unrealistically strong specular highlights, especially on soft materials or personal items. **MINIMIZE ALL GLOSSINESS AND REFLECTIVE EFFECTS.** **The surface of added objects should be MATTE or softly worn.**
- The added objects MUST be structurally complete, intact items (even if crumpled or torn). **Ensure bottles, cups, or containers have a single, coherent structure and do not exhibit impossible features like two openings.**
- Especially for fabric or leather items, ensure realistic surface texture and material detail, avoiding a flat or overly smooth appearance.

Lighting and blending:
- Match the original lighting direction, color, and contrast of the photo.
- **Ensure new objects are correctly affected by all ambient shadows and cast shadows originating from the bike frame and surrounding environment, not just contact shadows.**
- **Significantly reduce the saturation and sharpness of the added items to perfectly blend with the low-resolution, grainy, compressed background.**
- Add realistic, dark, and soft contact shadows where objects touch the basket or floor. **The shadows must deeply anchor the items and eliminate any "pasted" appearance.**
- Slightly soften the edges of new objects so they blend smoothly into the scene. Crucially, avoid harsh cut-out borders or sharp outlines around the added items.

Do NOT:
- Move or deform the metal basket, frame, wheel, or existing cables.
- **Paint over the original metal grid wires, or place added objects in front of the wires.**
- **Extend any added object's silhouette beyond the physical boundaries or metal frame of the basket.**
- Add people, hands, faces, or animals.
- Add text, brand logos, icons, user interface elements, or watermarks.
- Change the style to cartoon, anime, or 3D render.
""".strip()


def build_prompt(content_type: str, intensity: int, damage: str) -> str:
    """
    content_type + intensity + damage + txt 기반 object 예시를 합쳐
    최종 프롬프트 문자열 생성.
    """
    base = CONTENT_PHRASES[content_type]
    intensity_phrase = intensity_to_phrase(intensity)

    # txt에서 예시 객체 이름 샘플링 (light/medium/personal 만)
    example_phrase = ""
    if content_type in ("light_trash", "medium_trash", "personal_items"):
        example_phrase = sample_object_examples(content_type, intensity)

    if example_phrase:
        base = (
            base
            + " Use realistic objects such as "
            + example_phrase
            + "."
        )

    if damage == "none":
        damage_sentence = "Keep the overall cleanliness of the basket similar to the original photo."
    else:
        damage_sentence = DAMAGE_PHRASES[damage]

    return PROMPT_TEMPLATE.format(
        content_sentence=base,
        intensity_phrase=intensity_phrase,
        damage_sentence=damage_sentence,
    )


# ===============================
# 8. 응답 → PIL.Image
# ===============================

def part_to_image(part) -> Image.Image | None:
    inline = getattr(part, "inline_data", None)
    if inline is None:
        return None
    data = getattr(inline, "data", None)
    if data is None:
        return None
    return Image.open(BytesIO(data))


# ===============================
# 9. Gemini 호출
# ===============================

def edit_image_with_gemini(image_path: str, prompt: str, output_path: str) -> bool:
    image_input = Image.open(image_path)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_input],
        )
    except errors.ClientError as e:
        print(f"❌ Gemini ClientError: {e}")
        return False
    except Exception as e:
        print(f"❌ Gemini 기타 오류: {e}")
        return False

    # candidates → content.parts 쪽에서 이미지 찾기
    for cand in getattr(response, "candidates", []):
        content = getattr(cand, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []):
            img = part_to_image(part)
            if img is not None:
                img.save(output_path)
                print(f"✅ Saved: {output_path}")
                return True

    # fallback: response.parts 직접
    if hasattr(response, "parts"):
        for part in response.parts:
            img = part_to_image(part)
            if img is not None:
                img.save(output_path)
                print(f"✅ Saved (from response.parts): {output_path}")
                return True

    print(f"⚠️ 이미지 파트를 찾지 못했습니다 (텍스트 응답만 온 듯): {image_path}")
    return False


# ===============================
# 10. 배치 처리
# ===============================

def process_batch(batch_name: str):
    batch_dir = os.path.join(BATCHES_DIR, batch_name)
    input_dir = os.path.join(batch_dir, "crops")
    output_dir = os.path.join(batch_dir, "edits")

    if not os.path.isdir(input_dir):
        return

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    )
    if not files:
        return

    print("\n==============================")
    print(f"▶ 대상 폴더: {batch_dir}")
    print(f"▶ 입력 crops: {len(files)}장")

    sampled_cts = []

    for filename in files:
        in_path = os.path.join(input_dir, filename)

        content_type = sample_content_type()
        intensity = sample_intensity(content_type)
        damage = sample_damage(content_type)

        sampled_cts.append(content_type)

        out_name = f"edit_{content_type}_{intensity}_{damage}_{filename}"
        out_path = os.path.join(output_dir, out_name)

        print(f"\n[합성 시작] {filename}")
        print(f"  - content_type: {content_type}")
        print(f"  - intensity   : {intensity}")
        print(f"  - damage      : {damage}")

        # ✅ empty 인 경우: Gemini 호출 없이 원본을 그대로 복사 저장하고 패스
        if content_type == "empty":
            img = Image.open(in_path)
            img.save(out_path)
            print(f"  → empty basket: original image copied without editing → {out_path}")
            continue

        prompt = build_prompt(content_type, intensity, damage)

        ok = edit_image_with_gemini(in_path, prompt, out_path)
        if not ok:
            # 필요하면 실패 시 로직 조정
            pass

    print("\n=== 이 배치에서 샘플링된 content_type 분포 ===")
    print(Counter(sampled_cts))


def main():
    if not os.path.isdir(BATCHES_DIR):
        print(f"❌ batches 폴더가 없습니다: {BATCHES_DIR}")
        return

    batch_names = [
        d for d in os.listdir(BATCHES_DIR)
        if os.path.isdir(os.path.join(BATCHES_DIR, d))
    ]
    if not batch_names:
        print("❌ batches/ 하위 폴더가 없습니다.")
        return

    for batch_name in sorted(batch_names):
        process_batch(batch_name)


if __name__ == "__main__":
    main()

