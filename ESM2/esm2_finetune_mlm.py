import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from transformers import EsmForMaskedLM, AutoTokenizer


class ProteinDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            max_length=1024,
            padding="max_length",
            truncation=True,
        )
        labels = inputs.input_ids.clone()

        # Mask 15% of the tokens
        mask_indices = torch.bernoulli(torch.full(labels.shape, 0.15)).bool()
        inputs.input_ids[mask_indices] = self.tokenizer.mask_token_id

        # Create the inputs dictionary
        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        return inputs, labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Load model
# If you load the model using EsmForMaskedLM.from_pretrained("esm2_customized"),
# the model will be initialized with the updated parameters saved in the
# esm2_customized directory.
# If you load the model using EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D"),
# the model will be initialized with the original pre-trained parameters from the Hugging Face.

# More models:
# https://huggingface.co/facebook/esm2_t48_15B_UR50D
# Checkpoint name	Num layers	Num parameters
# esm2_t48_15B_UR50D	48	15B
# esm2_t36_3B_UR50D	36	3B
# esm2_t33_650M_UR50D	33	650M
# esm2_t30_150M_UR50D	30	150M
# esm2_t12_35M_UR50D	12	35M
# esm2_t6_8M_UR50D	6	8M


model_name = "facebook/esm2_t33_650M_UR50D"
model = EsmForMaskedLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sequence = "MKKFESRKIQMRRNILAKYANGK"

inputs = tokenizer(sequence, return_tensors="pt")

# Example sequences
sequences = ["MKKFESRKIQMRRNILAKYANGK", "MKKFESRKIQMRRNILAKYANGK"]

dataset = ProteinDataset(sequences, tokenizer)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch[0]["input_ids"].squeeze(1)  # Remove the extra dimension
        attention_mask = batch[0]["attention_mask"].squeeze(
            1
        )  # Remove the extra dimension
        labels = batch[1]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        optimizer.zero_grad()
        outputs = model(**inputs)

        loss = criterion(
            outputs.logits.view(-1, model.config.vocab_size), labels.view(-1)
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")

# Save the updated model to esm2_customized to the current working directory
model.save_pretrained("esm2_customized")
