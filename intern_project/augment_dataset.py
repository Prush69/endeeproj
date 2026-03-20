import json

def main() -> None:
    """
    Augments the offline ML papers dataset with accurate metadata (year, author)
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

    metadata_map = {
        "Attention Is All You Need": {"author": "Ashish Vaswani", "year": 2017},
        "BERT: Pre-training of Deep Bidirectional Transformers": {"author": "Jacob Devlin", "year": 2018},
        "Language Models are Few-Shot Learners": {"author": "Tom B. Brown", "year": 2020},
        "Mastering the game of Go with deep neural networks": {"author": "David Silver", "year": 2016},
        "Deep Residual Learning for Image Recognition": {"author": "Kaiming He", "year": 2015},
        "Generative Adversarial Nets": {"author": "Ian Goodfellow", "year": 2014},
        "Playing Atari with Deep Reinforcement Learning": {"author": "Volodymyr Mnih", "year": 2013},
        "Adam: A Method for Stochastic Optimization": {"author": "Diederik P. Kingma", "year": 2014},
        "YOLO9000: Better, Faster, Stronger": {"author": "Joseph Redmon", "year": 2016},
        "Distributed Representations of Words and Phrases and their Compositionality": {"author": "Tomas Mikolov", "year": 2013},
        "Evaluating Large Language Models Trained on Code": {"author": "Mark Chen", "year": 2021},
        "Training language models to follow instructions": {"author": "Long Ouyang", "year": 2022},
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks": {"author": "Patrick Lewis", "year": 2020},
        "A Survey of Vector Database Systems": {"author": "Hanwen Liu", "year": 2023}
    }

    print(f"Enriching {len(docs)} documents with accurate metadata...")
    for doc in docs:
        title = doc.get("title", "")
        meta = metadata_map.get(title, {"author": "Unknown", "year": 2024})
        doc["year"] = meta["year"]
        doc["author"] = meta["author"]

    print(f"Saving enriched dataset to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(docs, f, indent=4)

    print("Dataset augmentation complete!")

if __name__ == "__main__":
    main()
