import os

ROOT = "data/testdata/semantic-test-dataset"

def crawl(folder):
    for root, dirs, files in os.walk(folder):
        print(f"root {root}")
        print(f"dirs {dirs}")
        print(f"files {files}")

if __name__ == "__main__":
    crawl(ROOT)


