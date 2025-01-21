from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.samples = [
            {"prompt": "The sun rises in the", "response": "east."},
            {"prompt": "Water boils at", "response": "100 degrees Celsius."},
        ]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = self.samples[idx]
        input_text = f"Q: {data['prompt']} A: "
        target_text = data["response"]
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        targets = self.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True)
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0)
        }
