"""Rewrite every raw H&M caption into alt-text with a text-only LLM (Qwen2.5-7B).

    python scripts/02_rewrite_captions.py                       # all captions
    python scripts/02_rewrite_captions.py --limit 300 --out data/alt_text_trial.jsonl

Every rewrite is checked. A rewrite that fails is sent back with the reason, up to twice.
If it still fails it is saved with ok=false and not used for training or evaluation.
Run it again after an interruption and it continues where it stopped.
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import ALT_TEXT_TARGET_CHARS, DATA_DIR, OUTPUT_DIR, REWRITER_REPO
from utils.io import append_jsonl, read_jsonl
from utils.textnorm import rewrite_issues, split_caption, tidy_output

SEED = 0
MAX_RETRIES = 2

SYSTEM_PROMPT = f"""You write alt text for retail clothing product photos, for people who use \
screen readers. For one garment you are given three catalogue fields: a pattern label, a colour \
and a description. Rewrite them as alt text.

Rules:
1. Write one sentence in sentence case, ending with a period. Keep it to at most \
{ALT_TEXT_TARGET_CHARS} characters.
2. Start with the colour, then the pattern if there is one, then the garment type.
3. Pattern label: "solid" means one plain colour, so write only the colour. Any other label \
means the garment is not plain and the alt text must say so in natural words, for example \
"striped", "checked", "dotted", "patterned", "with a front print", "colour-block", "sequined". \
Ignore labels that describe nothing visible: "other structure", "treatment", "unknown".
4. Colour: if it is given as "other pink", write "pink". If it is only "other" or "unknown", do \
not state a colour.
5. Keep what a person could see in a photo of the front of the garment: garment type, length, \
neckline, collar, sleeves, straps, visible fastenings such as buttons or a zip, front pockets, \
trims and decorative details. Put the most distinctive details first and leave out minor ones \
when space is short.
6. Leave out everything that cannot be seen: fibre content (cotton, polyester, wool, viscose and \
so on), fillings, linings, anything inside the garment, concealed or hidden parts, sustainability \
claims (organic, recycled), how it feels (soft, stretchy), fit labels (regular fit, slim fit), \
function and comfort claims, denier, sizes, and anything supplied with the garment.
7. Leave out details that exist only at the back, such as a zip at the back, back pockets or a \
yoke at the back.
8. Use only facts from the three fields. Never add or guess a detail. Do not mention a brand, a \
collection, a person, or the words "image", "photo" or "picture".
9. Keep every detail attached to the part it belongs to. "long sleeves with buttoned cuffs" may \
stay as it is or become "long sleeves", but never "long buttoned sleeves".
10. The descriptions have lost their punctuation, so "34-length" means three-quarter length: \
write "three-quarter-length".

Reply with the alt text only."""

# real captions from the dataset with rewrites written by hand
FEW_SHOT = [
    (
        "solid black jersey top with narrow shoulder straps",
        "Black jersey top with narrow shoulder straps.",
    ),
    (
        "melange dark grey trousers in sweatshirt fabric with an elasticated drawstring waist "
        "side pockets a back pocket and ribbed hems soft brushed inside",
        "Dark grey melange sweatshirt trousers with an elasticated drawstring waist, side pockets "
        "and ribbed hems.",
    ),
    (
        "solid black short padded jacket with a jersey-lined hood and stand-up collar with a "
        "drawstring zip down the front flap side pockets with a press-stud one inner pocket and "
        "ribbing at the cuffs and hem quilted lining",
        "Black short padded jacket with a hood, stand-up collar, front zip, flap side pockets and "
        "ribbed cuffs and hem.",
    ),
    (
        "all over pattern dark green calf-length skirt in a patterned viscose crepe weave with a "
        "high waist and concealed zip in one side unlined",
        "Dark green patterned calf-length skirt with a high waist.",
    ),
    (
        "stripe dark blue shirt in a striped premium cotton weave with a turn-down collar french "
        "front long sleeves with adjustable buttoning at the cuffs and a rounded hem slim fit "
        "with narrow shoulders and a tapered waist for a fitted silhouette",
        "Dark blue striped shirt with a turn-down collar, long sleeves with buttoned cuffs and a "
        "rounded hem.",
    ),
    (
        "solid black 50 denier tights with reinforcement at the top for a shaping effect on the "
        "tummy and thighs",
        "Black tights with a reinforced top.",
    ),
    (
        "placement print black sports top in fast-drying functional fabric with a straight-cut "
        "front and long sleeves with a print motif at the cuffs rounded and longer at the back",
        "Black long-sleeved sports top with a straight-cut front and a print motif at the cuffs.",
    ),
    (
        "solid light blue dungarees in washed stretch denim with a chest pocket and adjustable "
        "straps with metal fasteners fake fly a zip in one side patch front pockets back pockets "
        "and slim legs",
        "Light blue washed denim dungarees with a chest pocket, adjustable straps with metal "
        "fasteners and patch front pockets.",
    ),
    (
        "check dark red double-breasted jacket in boucle made from a cotton blend with notch "
        "lapels and flap front pockets lined",
        "Dark red checked double-breasted jacket with notch lapels and flap front pockets.",
    ),
    (
        "front print black short straight dress in soft cotton jersey with a print motif on the "
        "front ribbed trim around the neckline dropped shoulders and short sleeves",
        "Black short straight jersey dress with a print motif on the front, dropped shoulders and "
        "short sleeves.",
    ),
    (
        "solid other orange boxy cardigan knitted in a soft wool blend with a v-neck large "
        "buttons down the front long raglan sleeves and wide ribbing at the cuffs and hem",
        "Orange boxy knitted cardigan with a V-neck, large buttons down the front, long raglan "
        "sleeves and wide ribbed cuffs and hem.",
    ),
    (
        "lace dark blue fitted knee-length dress in lace with a small stand-up collar concealed "
        "zip at the back cap sleeves and an inset lace trim at the waist lined",
        "Dark blue fitted knee-length lace dress with a small stand-up collar, cap sleeves and a "
        "lace trim at the waist.",
    ),
    (
        "embroidery grey jumper in a fine airy knit with embroidery on the front and 34-length "
        "sleeves with ribbed flounces at the cuffs",
        "Grey fine-knit jumper with embroidery on the front and three-quarter-length sleeves with "
        "ribbed flounces at the cuffs.",
    ),
    (
        "all over pattern beige wide shorts in woven fabric with a sheen high waist and creases "
        "zip fly with a hook-and-eye fastening side pockets and fake welt back pockets",
        "Beige patterned wide shorts with a sheen, a high waist, creases and side pockets.",
    ),
]


def as_fields(raw: str) -> str:
    parts = split_caption(raw)
    return (f"Pattern label: {parts['pattern'] or 'unknown'}\n"
            f"Colour: {parts['colour'] or 'unknown'}\n"
            f"Description: {parts['body']}")


def build_messages(raw: str, previous: str | None = None, issues: list[str] | None = None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for source, target in FEW_SHOT:
        messages.append({"role": "user", "content": as_fields(source)})
        messages.append({"role": "assistant", "content": target})
    messages.append({"role": "user", "content": as_fields(raw)})
    if previous is not None:
        shorten = any(issue.startswith("too long") for issue in issues)
        messages.append({"role": "assistant", "content": previous})
        messages.append(
            {
                "role": "user",
                "content": "That answer was rejected because it " + "; ".join(issues) + ". "
                "Write the alt text again with that fixed"
                + (", in at most 115 characters" if shorten else " and everything else unchanged")
                + ". Keep it a complete, natural sentence. Reply with the alt text only.",
            }
        )
    return messages


@torch.inference_mode()
def generate(model, tokenizer, conversations, max_new_tokens=96) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in conversations
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy, so a rerun gives the same text
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output[:, encoded["input_ids"].shape[1]:]  # drop the prompt, keep the answer
    return [tidy_output(t) for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def issues_for(raw: str, text: str) -> list[str]:
    return rewrite_issues(text, split_caption(raw)["pattern"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DATA_DIR / "alt_text.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="rewrite a random subset (trial run)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    for source, target in FEW_SHOT:  # the examples must pass the checks themselves
        assert not issues_for(source, target) and len(target) <= ALT_TEXT_TARGET_CHARS, target

    rows = read_jsonl(DATA_DIR / "meta.jsonl")
    if args.limit:
        rows = random.Random(SEED).sample(rows, args.limit)

    out_path = Path(args.out)
    # a job killed at its time limit can leave a half-written last line
    if out_path.exists():
        lines = out_path.read_text().splitlines(keepends=True)
        if lines and not lines[-1].rstrip().endswith("}"):
            out_path.write_text("".join(lines[:-1]))
            print("dropped an incomplete last line left by an interrupted run", flush=True)
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    todo.sort(key=lambda r: len(r["raw_caption"]))  # similar lengths in a batch = less padding
    print(f"{len(rows)} captions, {len(done)} already done, {len(todo)} to rewrite", flush=True)

    if todo:
        tokenizer = AutoTokenizer.from_pretrained(REWRITER_REPO)
        tokenizer.padding_side = "left"  # batched generation needs the padding on the left
        model = AutoModelForCausalLM.from_pretrained(REWRITER_REPO, dtype=torch.bfloat16)
        model.to("cuda").eval()

    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        texts = generate(model, tokenizer, [build_messages(r["raw_caption"]) for r in batch])
        results = [
            {"id": r["id"], "raw_caption": r["raw_caption"], "alt_text": t, "attempts": 1,
             "issues": issues_for(r["raw_caption"], t)}
            for r, t in zip(batch, texts)
        ]
        # send the failed ones back together with the reason they failed
        for _ in range(MAX_RETRIES):
            failed = [res for res in results if res["issues"]]
            if not failed:
                break
            retries = generate(
                model, tokenizer,
                [build_messages(res["raw_caption"], res["alt_text"], res["issues"]) for res in failed],
            )
            for res, text in zip(failed, retries):
                res.setdefault("first_attempt", res["alt_text"])
                res["alt_text"], res["attempts"] = text, res["attempts"] + 1
                res["issues"] = issues_for(res["raw_caption"], text)
        for res in results:
            res["ok"] = not res["issues"]
        append_jsonl(out_path, results)
        if (start // args.batch_size) % 20 == 0:
            print(f"  {start + len(batch)}/{len(todo)}", flush=True)

    # summary, plus two files for checking the rewrites by hand
    wanted = {row["id"] for row in rows}
    final = [r for r in read_jsonl(out_path) if r["id"] in wanted]
    ok = [r for r in final if r["ok"]]
    lengths = sorted(len(r["alt_text"]) for r in ok)
    reasons = Counter(issue.split(" (")[0].split(":")[0] for r in final for issue in r["issues"])
    print("\nSummary")
    print(f"  rewritten                 {len(final)}")
    print(f"  passed every check        {len(ok)} ({len(ok) / len(final):.1%})")
    print(f"  needed a retry            {sum(r['attempts'] > 1 for r in final)}")
    print(f"  median length (chars)     {lengths[len(lengths) // 2]}")
    print(f"  within {ALT_TEXT_TARGET_CHARS} chars          "
          f"{sum(n <= ALT_TEXT_TARGET_CHARS for n in lengths) / len(lengths):.1%}")
    for reason, count in reasons.most_common():
        print(f"  rejected: {reason:<45} {count}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.limit is None else "_trial"
    sample = random.Random(SEED).sample(ok, min(100, len(ok)))
    pd.DataFrame(sample)[["id", "raw_caption", "alt_text"]].to_csv(
        OUTPUT_DIR / f"rewrite_spotcheck{suffix}.csv", index=False)
    rejected = [r for r in final if not r["ok"]]
    if rejected:
        pd.DataFrame(rejected)[["id", "raw_caption", "alt_text", "issues"]].to_csv(
            OUTPUT_DIR / f"rewrite_rejected{suffix}.csv", index=False)
    print(f"  spot-check sample -> outputs/rewrite_spotcheck{suffix}.csv")


if __name__ == "__main__":
    main()
