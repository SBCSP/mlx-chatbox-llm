# from llama_index.llms.mlx import MLXLLM

# llm = MLXLLM(
#     model_name="microsoft/phi-2",
#     context_window=3900,
#     max_new_tokens=256,
#     # Remove generate_kwargs for now to test basic functionality
#     # generate_kwargs={"temperature": 0.7, "top_p": 0.95},
# )

# response = llm.complete("What is the meaning of life?")
# print(str(response))

from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache

# Load a 4-bit quantized Mistral 7B model 
model, tokenizer = load(
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    tokenizer_config={"trust_remote_code": True}
)
# Generate text from a prompt
prompt = "Write a haiku about coding on a rainy day."
response = generate(
    model, 
    tokenizer, 
    prompt, 
    max_tokens=256,
    verbose=True  # Shows generation stats like speed
)
print(response)