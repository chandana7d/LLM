def build_vocab(preprocessed_text):
    """
    Build a vocabulary from the tokenized text.
    :param preprocessed_text: List of preprocessed text tokens.
    :return: Dictionary mapping tokens to integer IDs and vocabulary size.
    """
    # Sort and get unique tokens
    all_tokens = sorted(list(set(preprocessed_text)))
    # Add special tokens for unknown and end-of-text cases
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    # Create a vocabulary with token-ID mappings
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    vocab_size = len(vocab)
    return vocab, vocab_size


def preprocess_text(raw_text):
    """
    Preprocess raw text by tokenizing and cleaning it.
    :param raw_text: Raw input string.
    :return: List of cleaned and tokenized text.
    """
    import re
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    return [item.strip() for item in preprocessed if item.strip()]
