import json
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from docx import Document


# AI generated script to create a test volume for testing crawling and rerieval


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "data/testdata/semantic-test-dataset"

CATEGORY_MAPPING = {
    "boat": "boats",
    "airplane": "airplanes",
    "car": "cars",
    "dog": "animals"
}

IMAGES_PER_CATEGORY = 10
TEXTS_PER_CATEGORY = 10


# ============================================================
# Verify COCO files exist
# ============================================================

def verify_coco():
    if not (DATA_DIR / "val2017").exists():
        raise RuntimeError("Missing data/val2017 folder.")
    if not (DATA_DIR / "annotations" / "instances_val2017.json").exists():
        raise RuntimeError("Missing instances_val2017.json.")
    print("[OK] COCO files verified.")


# ============================================================
# Folder structure
# ============================================================

def create_structure():
    for folder in CATEGORY_MAPPING.values():
        (OUTPUT_DIR / folder / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / folder / "texts").mkdir(parents=True, exist_ok=True)


# ============================================================
# Image processing
# ============================================================

def resize_image(src, dst):
    img = Image.open(src).convert("RGB")
    img.thumbnail((800, 800))
    img.save(dst, quality=90)


# ============================================================
# Meaningful Text Generator
# ============================================================

def generate_texts(category_name):
    base_descriptions = {
        "boats": [
            "Boats are watercraft designed for travel across rivers, lakes, and oceans.",
            "Modern boats range from small fishing vessels to large cargo ships.",
            "Sailboats rely on wind power, while motorboats use combustion engines.",
            "Boats are commonly used for recreation, transport, and maritime trade.",
            "Harbors and marinas are infrastructure environments for boats.",
            "Naval engineering focuses on hull design and buoyancy principles.",
            "Passenger boats are used in tourism and public transportation.",
            "Some boats are built specifically for racing competitions.",
            "Fishing boats operate in both coastal and deep-sea environments.",
            "Cargo ships transport goods between international ports."
        ],
        "airplanes": [
            "Airplanes are fixed-wing aircraft used for air transportation.",
            "Commercial airplanes transport passengers across continents.",
            "Jet engines allow airplanes to reach high cruising speeds.",
            "Airplanes require runways for takeoff and landing.",
            "Cargo aircraft are designed to transport freight by air.",
            "Military airplanes are used for defense and reconnaissance.",
            "Air travel connects cities across the globe.",
            "Aircraft engineering focuses on aerodynamics and lift.",
            "Propeller airplanes are often used for regional flights.",
            "Modern airplanes rely heavily on advanced navigation systems."
        ],
        "cars": [
            "Cars are motor vehicles designed for road transportation.",
            "Electric cars use battery-powered propulsion systems.",
            "Sports cars are optimized for high speed and acceleration.",
            "Cars are central to personal mobility in urban areas.",
            "Autonomous cars rely on sensors and machine learning.",
            "Family cars prioritize safety and passenger comfort.",
            "Hybrid cars combine combustion engines with electric motors.",
            "Cars operate on highways, streets, and rural roads.",
            "Vehicle design includes engine, chassis, and suspension systems.",
            "Car manufacturing is a major global industry."
        ],
        "animals": [
            "Animals are living organisms capable of movement and perception.",
            "Dogs are domesticated animals often kept as pets.",
            "Wild animals inhabit forests, grasslands, and oceans.",
            "Animal behavior is studied in zoology.",
            "Mammals are warm-blooded vertebrate animals.",
            "Animals play important roles in ecosystems.",
            "Some animals are used in agricultural environments.",
            "Wildlife conservation protects endangered species.",
            "Animal habitats vary from urban to remote regions.",
            "Domesticated animals assist humans in various tasks."
        ]
    }

    return base_descriptions[category_name]


# ============================================================
# Save texts in multiple formats
# ============================================================

def save_text_variants(folder, category, texts):
    styles = getSampleStyleSheet()

    for idx, text in enumerate(texts):
        # TXT
        with open(folder / f"{category}_{idx}.txt", "w", encoding="utf-8") as f:
            f.write(text)

        # Markdown
        with open(folder / f"{category}_{idx}.md", "w", encoding="utf-8") as f:
            f.write(f"# {category.capitalize()}\n\n{text}")

        # PDF
        pdf = SimpleDocTemplate(str(folder / f"{category}_{idx}.pdf"))
        pdf.build([Paragraph(text, styles["Normal"])])

        # DOCX
        doc = Document()
        doc.add_heading(category.capitalize(), 1)
        doc.add_paragraph(text)
        doc.save(folder / f"{category}_{idx}.docx")


# ============================================================
# Main builder
# ============================================================

def build_dataset():
    print("Building compact semantic retrieval dataset (10x10)...")

    verify_coco()
    create_structure()

    with open(DATA_DIR / "annotations" / "instances_val2017.json") as f:
        coco = json.load(f)

    id_to_cat = {c["id"]: c["name"] for c in coco["categories"]}
    id_to_img = {img["id"]: img["file_name"] for img in coco["images"]}

    counters = {folder: 0 for folder in CATEGORY_MAPPING.values()}
    used_images = set()

    for ann in tqdm(coco["annotations"]):
        cat_name = id_to_cat[ann["category_id"]]

        if cat_name in CATEGORY_MAPPING:
            target = CATEGORY_MAPPING[cat_name]

            if counters[target] >= IMAGES_PER_CATEGORY:
                continue

            img_id = ann["image_id"]
            if img_id in used_images:
                continue

            filename = id_to_img[img_id]
            src = DATA_DIR / "val2017" / filename
            dst = OUTPUT_DIR / target / "images" / filename

            resize_image(src, dst)

            counters[target] += 1
            used_images.add(img_id)

        if all(v >= IMAGES_PER_CATEGORY for v in counters.values()):
            break

    # Generate texts
    for category in CATEGORY_MAPPING.values():
        texts = generate_texts(category)
        save_text_variants(OUTPUT_DIR / category / "texts", category, texts)

    print("Dataset successfully built.")
    print("10 images and 10 texts per category.")
    

if __name__ == "__main__":
    build_dataset()
