import sys

from transformers import GPT2LMHeadModel, GPT2Tokenizer

out = sys.argv[1] if len(sys.argv) > 1 else "models"

model = GPT2LMHeadModel.from_pretrained("gpt2")
for name, param in model.named_parameters():
    print(name, tuple(param.shape))
    with open(f"{out}/{name}.txt", "w") as f:
        f.write(" ".join(str(v) for v in param.detach().numpy().flatten()))

GPT2Tokenizer.from_pretrained("gpt2").save_pretrained(f"{out}/tokenizer")
