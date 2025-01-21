from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
import torch

from toy_dataset import ToyDataset
from load_model import load_model

model, tokenizer = load_model()
dataset = ToyDataset(tokenizer)

# LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
