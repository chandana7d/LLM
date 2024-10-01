import os
import urllib.request
from tokenizer import SimpleTokenizerV1
from vocab_builder import build_vocab, preprocess_text


def download_text(file_name, url):
    """
    Download a text file if it doesn't already exist.
    :param file_name: Name of the file to save as.
    :param url: URL of the text file to download.
    """
    if not os.path.exists(file_name):
        urllib.request.urlretrieve(url, file_name)
        print(f"Downloaded {file_name}")
    else:
        print(f"{file_name} already exists.")


def load_text(file_name):
    """
    Load text from a file.
    :param file_name: Name of the file to read from.
    :return: The contents of the file as a string.
    """
    with open(file_name, "r", encoding="utf-8") as f:
        return f.read()


def main():
    # Download the text if not present
    file_path = "data/the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    download_text(file_path, url)

    # Load and preprocess the text
    raw_text = load_text(file_path)
    preprocessed_text = preprocess_text(raw_text)

    # Build the vocabulary
    vocab, vocab_size = build_vocab(preprocessed_text)
    print(f"Vocabulary size: {vocab_size}")

    # Initialize the tokenizer with the vocabulary
    tokenizer = SimpleTokenizerV1(vocab)

    # Test encoding and decoding
    sample_text = """It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
    token_ids = tokenizer.encode(sample_text)
    print("Encoded:", token_ids)

    decoded_text = tokenizer.decode(token_ids)
    print("Decoded:", decoded_text)


if __name__ == "__main__":
    main()
