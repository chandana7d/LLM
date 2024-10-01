import torch
from torch.utils.data import Dataset, DataLoader


def create_number_data_file(filename="data/number-data.txt", upper_limit=1000):
    """Create a text file containing numbers from 0 to upper_limit."""
    with open(filename, "w", encoding="utf-8") as f:
        for number in range(upper_limit + 1):
            f.write(f"{number} ")


class GPTDatasetV1(Dataset):
    """Dataset class for GPT-style models using a sliding window approach."""

    def __init__(self, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Parse integers directly from the text file
        token_ids = [int(i) for i in txt.strip().split()]

        # Use a sliding window to chunk the data into overlapping sequences
        for i in range(0, len(token_ids) - max_length):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    """Create a DataLoader for the GPTDatasetV1."""
    # Create dataset
    dataset = GPTDatasetV1(txt, max_length, stride)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


def test_dataloader(raw_text):
    """Test the DataLoader with the raw text input."""
    print("\nTesting DataLoader with batch size of 1:")
    dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

    data_iter = iter(dataloader)
    for _ in range(3):  # Show the first three batches
        inputs, targets = next(data_iter)
        print("Inputs:\n", inputs)
        print("Targets:\n", targets)

    # Get the last batch
    for batch in dataloader:
        pass

    last_batch = batch
    print("Last Batch Inputs:\n", last_batch[0])
    print("Last Batch Targets:\n", last_batch[1])

    # Show inputs and targets with batch size of 2
    print("\nTesting DataLoader with batch size of 2:")
    dataloader = create_dataloader(raw_text, batch_size=2, max_length=4, stride=4, shuffle=False)

    for inputs, targets in dataloader:
        print("Inputs:\n", inputs)
        print("Targets:\n", targets)

    # DataLoader with shuffling
    print("\nTesting DataLoader with shuffling:")
    torch.manual_seed(123)  # Set seed for reproducibility
    dataloader = create_dataloader(raw_text, batch_size=2, max_length=4, stride=4, shuffle=True)

    for inputs, targets in dataloader:
        print("Inputs (Shuffled):\n", inputs)
        print("Targets (Shuffled):\n", targets)


def main():
    """Main function to execute the program."""
    # Create a text file with number data
    create_number_data_file()

    # Load the number data
    with open("data/number-data.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Test the DataLoader
    test_dataloader(raw_text)


if __name__ == "__main__":
    main()
