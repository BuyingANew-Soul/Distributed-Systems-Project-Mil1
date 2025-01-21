from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load fine-tuned model
MODEL_PATH = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

class QueryRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(request: QueryRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=50)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
