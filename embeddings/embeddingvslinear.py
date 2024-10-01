import torch
import torch.nn.functional as F
import os


# Class to encapsulate the Embedding layer logic
class EmbeddingModel:
    def __init__(self, num_tokens, embedding_dim):
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.embedding_layer = torch.nn.Embedding(num_tokens, embedding_dim)
        torch.manual_seed(123)  # For reproducibility

    def forward(self, token_ids):
        return self.embedding_layer(token_ids)

    def get_weights(self):
        return self.embedding_layer.weight


# Class to encapsulate the Linear layer logic with one-hot encoding
class LinearModel:
    def __init__(self, num_tokens, embedding_dim):
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.linear_layer = torch.nn.Linear(num_tokens, embedding_dim, bias=False)
        torch.manual_seed(123)  # For reproducibility

    def forward(self, one_hot_encoded_input):
        return self.linear_layer(one_hot_encoded_input.float())

    def set_weights(self, embedding_weights):
        # Set weights from the embedding layer
        self.linear_layer.weight = torch.nn.Parameter(embedding_weights.T)


# Class to handle comparison between Embedding and Linear Layer
class EmbeddingVsLinear:
    def __init__(self, token_ids, embedding_dim):
        self.token_ids = token_ids
        self.embedding_dim = embedding_dim
        self.num_tokens = max(token_ids) + 1  # Number of unique token IDs

        # Initialize the models
        self.embedding_model = EmbeddingModel(self.num_tokens, embedding_dim)
        self.linear_model = LinearModel(self.num_tokens, embedding_dim)

        # Set the same weights for Linear as Embedding for direct comparison
        self.linear_model.set_weights(self.embedding_model.get_weights())

    def compare_results(self):
        # Get embeddings from the Embedding layer
        embedding_output = self.embedding_model.forward(self.token_ids)

        # Convert token IDs to one-hot encoding
        one_hot_input = F.one_hot(self.token_ids, num_classes=self.num_tokens)
        print(one_hot_input)

        # Get output from Linear layer using one-hot encoding
        linear_output = self.linear_model.forward(one_hot_input)

        # Compare if the outputs are identical
        are_equal = torch.allclose(embedding_output, linear_output, atol=1e-6)

        # Store results in a dictionary for easy writing to file
        results = {
            "Embedding Layer Output": embedding_output,
            "Linear Layer Output": linear_output,
            "Are the outputs the same?": "Yes" if are_equal else "No"
        }
        return results

    def save_results_to_file(self, results, file_path):
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}:\n{value}\n\n")


# Main function
def main():
    # Define token IDs and embedding dimension
    token_ids = torch.tensor([2, 3, 1])
    embedding_dim = 5

    # Initialize the comparison module
    comparison = EmbeddingVsLinear(token_ids, embedding_dim)

    # Perform the comparison
    results = comparison.compare_results()

    # Specify the output file path
    output_file_path = 'output/embeddingvslinear_comparison_results.txt'

    # Save results to the output file
    comparison.save_results_to_file(results, output_file_path)

    print(f"Comparison results saved to {output_file_path}")


# Ensure this script runs only when executed as the main program
if __name__ == "__main__":
    main()
