import json
from pathlib import Path

import random
random.seed(161)

OUTPUT_PATH = Path("data/testdata/documents.json")

# synthetic document dummies clustered by categories for testing if similar data are close together in final index
CLUSTERS = {
    "dogs": [
        "dog",
        "puppy",
        "barking dog",
        "dog training",
        "pet dog",
        "canine",
        "dog park",
        "dog leash",
    ],
    "cats": [
        "cat",
        "kitten",
        "meowing cat",
        "pet cat",
        "feline",
        "cat sleeping",
        "cat food",
    ],
    "cars": [
        "car",
        "electric car",
        "sports car",
        "engine",
        "vehicle",
        "car driving",
        "car maintenance",
    ],
    "programming": [
        "programming",
        "python code",
        "software development",
        "algorithms",
        "data structures",
        "debugging code",
        "backend system",
    ],
    "music": [
        "music",
        "guitar",
        "piano",
        "concert",
        "listening to music",
        "audio recording",
        "song melody",
    ],
}

def generate_documents():
    documents = []
    doc_counter = 1

    for cluster_name, phrases in CLUSTERS.items():
        for phrase in phrases:
            documents.append({
                "id": f"doc_{doc_counter:03d}",
                "modality": "text",
                "metadata": {
                    "source": "synthetic",
                    "cluster": cluster_name,
                    "synthetic_phrase": phrase
                    # phrases are dummy data that later is replaced with a vector from vectorising image, text etc. data
                },
                "reference": {
                    "type": "dummy"
                }
            })
            doc_counter += 1

    random.shuffle(documents)
    return documents

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    documents = generate_documents()

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    print(f"Generated {len(documents)} documents at {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
