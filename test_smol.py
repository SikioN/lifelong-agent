from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
prompt = "Action:\n"
cand = "ACTION_0"
prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
full_ids = tokenizer(prompt + cand, add_special_tokens=False)["input_ids"]
cand_ids = full_ids[len(prompt_ids):]
print("cand_ids:", cand_ids)
print("decoded cand_ids:", tokenizer.decode(cand_ids))
