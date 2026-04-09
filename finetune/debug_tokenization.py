"""Debug script: inspect tokenization to find why response template isn't matched."""

import json
from transformers import AutoTokenizer

HF_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DATASET = "dataset_nav4rail_llm_2000.jsonl"

print(f"Loading tokenizer: {HF_ID}")
tokenizer = AutoTokenizer.from_pretrained(HF_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load first sample
with open(DATASET) as f:
    obj = json.loads(f.readline())
mission = obj["mission"].strip()
xml = obj["xml"].strip()

# Format with chat template
messages = [
    {"role": "system", "content": "SYS_PROMPT"},
    {"role": "user", "content": f"Mission : {mission}"},
    {"role": "assistant", "content": xml},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)

print(f"\n=== Chat template text (first 500 chars) ===")
print(repr(text[:500]))

# Show the boundary area around "assistant"
idx = text.rfind("assistant")
if idx >= 0:
    print(f"\n=== Text around last 'assistant' (pos {idx}) ===")
    print(repr(text[idx - 50 : idx + 80]))

# Tokenize with different settings
print(f"\n=== Tokenization comparison ===")
ids_no_special = tokenizer(text, add_special_tokens=False)["input_ids"]
ids_with_special = tokenizer(text, add_special_tokens=True)["input_ids"]
ids_chat = tokenizer.apply_chat_template(messages, tokenize=True)

print(
    f"add_special_tokens=False: {len(ids_no_special)} tokens, first 10: {ids_no_special[:10]}"
)
print(
    f"add_special_tokens=True:  {len(ids_with_special)} tokens, first 10: {ids_with_special[:10]}"
)
print(f"apply_chat_template:      {len(ids_chat)} tokens, first 10: {ids_chat[:10]}")


# Find "assistant" boundary in each
def find_boundary(ids, label):
    # Use a short test to find the assistant header
    test_msgs = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "RESP"},
    ]
    test_text = tokenizer.apply_chat_template(test_msgs, tokenize=False)
    test_ids_nospc = tokenizer(test_text, add_special_tokens=False)["input_ids"]
    test_ids_spc = tokenizer(test_text, add_special_tokens=True)["input_ids"]
    test_ids_chat = tokenizer.apply_chat_template(test_msgs, tokenize=True)

    resp_ids = tokenizer.encode("RESP", add_special_tokens=False)
    print(f"\n  [{label}] RESP token IDs: {resp_ids}")

    for name, tids in [
        ("no_special", test_ids_nospc),
        ("with_special", test_ids_spc),
        ("chat_template", test_ids_chat),
    ]:
        # Find RESP
        pos = -1
        for i in range(len(tids) - len(resp_ids), -1, -1):
            if tids[i : i + len(resp_ids)] == resp_ids:
                pos = i
                break
        if pos >= 0:
            before = tids[max(0, pos - 8) : pos]
            print(f"  [{name}] RESP at pos {pos}, 8 tokens before: {before}")
            print(f"    decoded before: {tokenizer.decode(before)!r}")
            # Try to find this "before" in the actual sample
            for j in range(len(ids) - len(before) + 1):
                if ids[j : j + len(before)] == before:
                    print(f"    FOUND in actual sample at pos {j}")
                    break
            else:
                print(f"    NOT FOUND in actual sample!")
                # Try subsets
                for n in range(len(before) - 1, 1, -1):
                    sub = before[-n:]
                    for j in range(len(ids) - len(sub) + 1):
                        if ids[j : j + len(sub)] == sub:
                            print(
                                f"    But {n}-token suffix found at pos {j}: {sub} = {tokenizer.decode(sub)!r}"
                            )
                            break
                    else:
                        continue
                    break
        else:
            print(f"  [{name}] RESP NOT FOUND in test!")

    return test_ids_nospc, test_ids_spc, test_ids_chat


find_boundary(ids_no_special, "no_special_actual")
find_boundary(ids_with_special, "with_special_actual")

# Also check what the collator would see
print(f"\n=== What SFTTrainer produces (simulated) ===")
# SFTTrainer tokenizes with defaults (add_special_tokens=True by default)
sft_ids = tokenizer(
    text,
    truncation=True,
    padding=False,
    max_length=8192,
    return_overflowing_tokens=False,
)["input_ids"]
print(f"SFTTrainer-like tokenization: {len(sft_ids)} tokens, first 10: {sft_ids[:10]}")

# Now compare with ids_no_special
if sft_ids == ids_no_special:
    print("SAME as add_special_tokens=False")
elif sft_ids == ids_with_special:
    print("SAME as add_special_tokens=True")
elif sft_ids == ids_chat:
    print("SAME as apply_chat_template")
else:
    print("DIFFERENT from all!")
    # Find first difference
    for i in range(min(len(sft_ids), len(ids_no_special))):
        if sft_ids[i] != ids_no_special[i]:
            print(
                f"  First diff at pos {i}: sft={sft_ids[i]} vs no_special={ids_no_special[i]}"
            )
            print(f"  sft context: {sft_ids[max(0, i - 3) : i + 3]}")
            print(f"  no_special context: {ids_no_special[max(0, i - 3) : i + 3]}")
            break
