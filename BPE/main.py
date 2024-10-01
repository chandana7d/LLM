from data import load_raw_text
from bpe_comparisons import compare_bpe_methods
from prettytable import PrettyTable


def main():
    # Load the raw text data
    raw_text = load_raw_text('data/the-verdict.txt')

    # Compare BPE methods
    results = compare_bpe_methods(raw_text)
    print(results.keys())

    # Create a comparison table
    table = PrettyTable()
    table.field_names = ["Method", "Encoded Tokens", "Decoding Time (ms)"]

    for method, (encoded, decoding_time) in results.items():
        table.add_row([method, len(encoded), f"{decoding_time:.5f}"])

    print(table)


if __name__ == "__main__":
    main()
