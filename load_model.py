# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM




def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
    return model, tokenizer


model = load_model()