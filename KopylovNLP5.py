import numpy as np 
import torch 
from transformers import GPT2LMHeadModel, GPT2Tokenizer 
np.random.seed(36) 
torch.manual_seed(36) 
 
def generate(model, tok, text, do_sample=True, max_length=120, repetition_penalty=1.0, top_k = 5, top_p=0.95, temperature = 1, num_beams=None, no_repeat_ngram_size=2): 
    input_ids = tok.encode(text, return_tensors="pt") 
    print(model.generate.__globals__['__file__']) 
    out = model.generate(input_ids, max_length = max_length, repetition_penalty = repetition_penalty, do_sample = do_sample, top_k = top_k, top_p = top_p, temperature = temperature, num_beams = num_beams, no_repeat_ngram_size = no_repeat_ngram_size) 
    return list(map(tok.decode, out)) 
 
prompt = 'я подошел к человеку, стоящему ко мне спиной, и' 
 
def load_tokenizer_and_model(model_name_or_path): 
    return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path) 
 
tok, model = load_tokenizer_and_model("sberbank-ai/rugpt3large_based_on_gpt2") 
generated = generate(model, tok, prompt, num_beams=10) 
print(generated[0])
