import os
import random
from io import BytesIO
from collections import Counter
import sys

import numpy as np
from PIL import Image

from google import genai
from google.genai import errors

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

MODEL_NAME = "gemini-3-pro-image-preview"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCHES_DIR = os.path.join(BASE_DIR, "batches")

CONFIG = {
    "empty": {
        "prob": 0.25,
        "min": 0,
        "max": 0,
        "mu": 0.0,
        "sigma": 0.0,
    },
    "light_trash": {
        "prob": 0.25,
        "min": 1,
        "max": 2,
        "mu": 2.0,
        "sigma": 1.0,
    },
    "medium_trash": {
        "prob": 0.25,
        "min": 1,
        "max": 2,
        "mu": 3.0,
        "sigma": 1.5,
    },
    "personal_items": {
        "prob": 0.25,
        "min": 1,
        "max": 2,
        "mu": 2.0,
        "sigma": 0.7,
    },
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

def intensity_to_phrase(intensity: int) -> str:
    if intensity == 1:
        return "one item"
    if intensity == 2:
        return "two items"
    return f"several items (around {intensity})"

from typing import Dict, List

OBJECT_ITEMS: Dict[str, List[str]] = {
    "light_trash": [
        "paper cup", "plastic bottle", "snack wrapper", "foil wrapper",
        "shopping receipt", "paper tissue", "plastic bag", "plastic lid",
        "coffee sleeve", "paper bag", "bottle cap", "straw", "cardboard scrap",
        "small cup lid", "chip wrapper", "snack crumbs", "plastic straw wrapper",
        "thin paper strip", "lightweight plastic piece", "small folded paper",
    ],
    "medium_trash": [
        "crushed plastic bottle", "aluminum can", "foil snack bag", "paper lunch bag",
        "mixed snack wrappers", "plastic cups", "large chip bag", "transparent container",
        "plastic takeout box", "plastic bag bundle", "paper tissues", "cardboard scraps",
        "foil strip", "coffee cup lid", "paper sleeve", "sandwich wrapper",
        "mixed paper scraps", "plastic fragments", "cardboard carrier piece", "foil twist",
    ],
    "personal_items": [
        "tumbler", "travel mug", "fabric pouch", "zipper pouch", "folded umbrella",
        "small shopping bag", "rolled reusable bag", "beanie hat", "notebook",
        "power bank", "camera pouch", "drawstring bag", "tablet sleeve", "scarf",
        "flat lunchbox", "cloth lunch wrap", "sports gloves", "folded picnic mat",
        "bottle", "wallet pouch",
    ],
}

def sample_object_examples(content_type: str, intensity: int) -> str:
    items = OBJECT_ITEMS.get(content_type, [])
    if not items:
        return ""
    k = max(1, min(intensity + 1, len(items)))
    chosen = random.sample(items, k=k)
    if len(chosen) == 1:
        return chosen[0]
    else:
        return ", ".join(chosen[:-1]) + ", and " + chosen[-1]

PROMPT_TEMPLATE = """
You are editing a cropped photo of a bicycle basket.
Use the input photo as the base and only add objects inside the basket.

Goals:
- Keep all original pixels of the metal basket wires, screws, bike frame, wheel, and background unchanged.
- Any new object must appear physically inside the basket, resting on the floor or stacked naturally.
- **CRITICAL: The inserted objects MUST EXACTLY match the low resolution, blurry texture, and noise level of the input image.**
- **New objects must be visually positioned behind the basket wires; never paint over the wires.**
- **The inserted objects should be placed near the top edge or upper half of the basket.**

Edit instructions:
- {add_instruction}
- The total amount of items should match this level: {intensity_phrase}.

Object appearance:
- Use realistic everyday objects, with varied but natural colors such as muted reds, greens, blues, and grays.
- Colors of different objects should not all be identical; allow subtle color diversity.
- Avoid unrealistically strong specular highlights, especially on soft materials or personal items.

Lighting and blending:
- Match the original lighting direction, color, and contrast of the photo.
- Add soft contact shadows where objects touch the basket or floor.
- Slightly soften the edges of new objects so they blend smoothly into the scene.
- Avoid harsh cut-out borders or halo artifacts around the added items.

Do NOT:
- Move or deform the metal basket, frame, wheel, or existing cables.
- Add people, hands, faces, or animals.
- Add text, brand logos, icons, user interface elements, or watermarks.
- Change the style to cartoon, anime, or 3D render.
- **Do NOT** introduce high-frequency details, sharpness, or clarity that exceeds the input image quality.
""".strip()

def build_prompt(content_type: str, intensity: int) -> str:
    intensity_phrase = intensity_to_phrase(intensity)

    if content_type == "empty":
        add_instruction = (
            "Do not add any new objects. Leave the inside of the basket completely empty."
        )
    else:
        example_phrase = ""
        if content_type in ("light_trash", "medium_trash", "personal_items"):
            example_phrase = sample_object_examples(content_type, intensity)

        if example_phrase:
            add_instruction = (
                "Add "
                + example_phrase
                + " inside the basket."
            )
        else:
            add_instruction = "Add a few small objects inside the basket."

    return PROMPT_TEMPLATE.format(
        add_instruction=add_instruction,
        intensity_phrase=intensity_phrase,
    )

def part_to_image(part) -> Image.Image | None:
    inline = getattr(part, "inline_data", None)
    if inline is None:
        return None
    data = getattr(inline, "data", None)
    if data is None:
        return None
    return Image.open(BytesIO(data))

def edit_image_with_gemini(image_path: str, prompt: str, output_path: str) -> bool:
    image_input = Image.open(image_path)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_input],
        )

    except errors.ClientError as e:
        # ✅ Gemini API 쪽에서 나는 모든 에러(403, 429 등)는 여기로 들어옴
        print(f"❌ Gemini ClientError: {e}")
        print("🚨 Gemini API 오류가 발생했습니다. 프로그램을 종료합니다.")
        sys.exit(1)   # 👉 에러 발생 시 바로 전체 프로그램 종료

    except Exception as e:
        # 그 외의 일반적인 예외 (파일 문제, PIL 에러 등)는 계속 False만 반환
        print(f"❌ Gemini 기타 오류: {e}")
        return False

    # 아래부터는 기존 로직 그대로
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

    if hasattr(response, "parts"):
        for part in response.parts:
            img = part_to_image(part)
            if img is not None:
                img.save(output_path)
                print(f"✅ Saved (from response.parts): {output_path}")
                return True

    print(f"⚠️ 이미지 파트를 찾지 못했습니다 (텍스트 응답만 온 듯): {image_path}")
    return False


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

        sampled_cts.append(content_type)

        out_name = f"edit_{content_type}_{intensity}_{filename}"
        out_path = os.path.join(output_dir, out_name)

        print(f"\n[합성 시작] {filename}")
        print(f"  - content_type: {content_type}")
        print(f"  - intensity   : {intensity}")

        if content_type == "empty":
            img = Image.open(in_path)
            img.save(out_path)
            print(f"  → empty basket: original image copied without editing → {out_path}")
            continue

        prompt = build_prompt(content_type, intensity)

        ok = edit_image_with_gemini(in_path, prompt, out_path)
        if not ok:
            print(f"⚠️ {filename} 편집 실패.")

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
