import json
import random

def main() -> None:
    """
    Augments the offline ML papers dataset with mock metadata (year, author)
    to enable testing of Endee's payload filtering capabilities.
    """
    input_file = "ml_papers.json"
    output_file = "ml_papers_enriched.json"

    print(f"Loading dataset from {input_file}...")
    try:
        with open(input_file, "r") as f:
            docs = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please ensure the dataset exists.")
        return

    authors = [
        "A. Vaswani", "Y. LeCun", "G. Hinton", "Y. Bengio",
        "I. Goodfellow", "K. He", "J. Redmon", "T. Chen",
        "A. Karpathy", "F. Chollet"
    ]

    print(f"Enriching {len(docs)} documents with mock metadata...")
    for doc in docs:
        doc["year"] = random.randint(2017, 2024)
        doc["author"] = random.choice(authors)

    print(f"Saving enriched dataset to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(docs, f, indent=4)

    print("Dataset augmentation complete!")

if __name__ == "__main__":
    main()
