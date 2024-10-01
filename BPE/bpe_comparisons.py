import time
import tiktoken
from bpe_openai_gpt2 import get_encoder, download_vocab
from transformers import GPT2Tokenizer

# Download vocab for the original BPE implementation
download_vocab()

# Initialize BPE implementations
tik_tokenizer = tiktoken.get_encoding("gpt2")
orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")
hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def compare_bpe_methods(raw_text):
    results = {}

    # Using tiktoken
    start_time = time.time()
    tik_integers = tik_tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
    tik_decoded = tik_tokenizer.decode(tik_integers)
    results['tiktoken'] = (tik_integers, (time.time() - start_time) * 1000)

    # Using the original BPE implementation
    start_time = time.time()
    orig_integers = orig_tokenizer.encode(raw_text)
    orig_decoded = orig_tokenizer.decode(orig_integers)
    results['original_bpe'] = (orig_integers, (time.time() - start_time) * 1000)

    # Using Hugging Face transformers
    start_time = time.time()
    hf_integers = hf_tokenizer.encode(raw_text, return_tensors="pt").tolist()[0]
    hf_decoded = hf_tokenizer.decode(hf_integers)
    results['hugging_face'] = (hf_integers, (time.time() - start_time) * 1000)

    return results

def load_raw_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    return raw_text

