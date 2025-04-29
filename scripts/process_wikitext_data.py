from datasets import load_dataset
import torch
from collections import Counter
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-v1")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

# Build vocabulary from training data
def build_vocab(data_iter):
    counter = Counter()
    for example in data_iter:
        tokens = tokenizer.tokenize(example["text"])
        counter.update(tokens)
    return {token: idx for idx, (token, _) in enumerate(counter.most_common())}

# Process text to tensor indices
def text_to_indices(data_iter, vocab):
    all_indices = []
    for example in data_iter:
        tokens = tokenizer.tokenize(example["text"])
        indices = [vocab.get(token, 0) for token in tokens]  # 0 for unknown
        all_indices.extend(indices)
    return torch.tensor(all_indices, dtype=torch.long)

# Build vocabulary from training data
vocab = build_vocab(dataset["train"])

# Convert splits to indices
train_data = text_to_indices(dataset["train"], vocab)
valid_data = text_to_indices(dataset["validation"], vocab)
test_data = text_to_indices(dataset["test"], vocab)

class DataContainer(torch.nn.Module):
    def __init__(self, data_dict):
        super().__init__()
        for key in data_dict:
            setattr(self, key, data_dict[key])

# Save processed data with TorchScript
container = DataContainer({
    "train": train_data,
    "valid": valid_data,
    "test": test_data,
    "vocab_size": torch.tensor(len(vocab))  # Convert to tensor
})
torch.jit.script(container).save("wikitext2_processed.pt")
