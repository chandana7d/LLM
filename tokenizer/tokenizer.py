import re


class SimpleTokenizerV1:
    def __init__(self, vocab):
        """
        Initialize the tokenizer with the given vocabulary.
        :param vocab: Dictionary mapping words/tokens to integer IDs.
        """
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        """
        Encode the input text into a list of integer token IDs.
        :param text: The input string.
        :return: List of token IDs.
        """
        # Tokenize and preprocess the input text
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Convert tokens into their corresponding IDs
        ids = [self.str_to_int.get(s, self.str_to_int.get('<|unk|>')) for s in preprocessed]
        return ids

    def decode(self, ids):
        """
        Decode a list of integer token IDs back into a string.
        :param ids: List of token IDs.
        :return: Decoded string.
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove extra spaces before punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
