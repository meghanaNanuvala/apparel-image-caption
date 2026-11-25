# Apparel Image Caption

Give it a retail clothing photo and it writes a short, objective description suitable for a
screen reader. Topic 2, base level: image captioning.

**Try it:** [demo on Hugging Face Spaces](https://huggingface.co/spaces/MeghanaNanuvala/apparel-image-caption) ·
[model on Hugging Face](https://huggingface.co/MeghanaNanuvala/apparel-image-caption)

The model is **Florence-2-base** (231M parameters, pretrained by Microsoft) fine-tuned on H&M
product images. The same test image, captioned three ways:

| System | Caption |
|---|---|
| **Fine-tuned (this project)** | Dark purple checked shirt with a turn-down collar, chest pockets, long sleeves with buttoned cuffs and a rounded hem. |
| Florence-2, zero-shot | A purple and black striped button down shirt. |
| BLIP, zero-shot | a picture of a purple and black plaid shirt |

## Results

970 held-out test images. No test image, and no other image sharing its description, was seen
in training.

| System | BLEU-4 | ROUGE-L | METEOR | CIDEr | CLIPScore | ≤125 chars | Sentence case + period | Mentions a person |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLIP-base, zero-shot | 2.3 | 22.1 | 18.0 | 18.5 | 77.9 | 97.8% | 0.0% | 9.7% |
| Florence-2-base, zero-shot (caption prompt) | 1.8 | 22.6 | 17.4 | 16.5 | 77.9 | 100.0% | 97.6% | 0.7% |
| Florence-2-base, zero-shot (detailed-caption prompt) | 0.0 | 10.2 | 8.3 | 5.3 | 65.4 | 100.0% | 100.0% | 0.0% |
| **Florence-2-base, fine-tuned** | **37.3** | **63.9** | **64.0** | **281.6** | **80.5** | 94.2% | 100.0% | 0.0% |
| *Reference alt-text (for context)* | | | | | 79.7 | 90.0% | 100.0% | 0.0% |
| *Original H&M caption (for context)* | | | | | 79.3 | 27.9% | 0.0% | 0.0% |

Scores are multiplied by 100. CIDEr's raw range is 0 to 10, so its maximum here is 1000. The full
table, including the remaining checks, is in `outputs/results.md`.

How to read this:

- **The reference metrics (BLEU to CIDEr) mostly measure style.** They reward wording close to the
  reference alt-text. The zero-shot models were never shown that style, so they score low by
  construction. The gap shows that fine-tuning taught the style; it does not by itself show the
  captions are more accurate.
- **CLIPScore is the fairer comparison**, because it scores the caption against the image and
  needs no reference. The fine-tuned model gains 2.6 points over both baselines. In a paired
  comparison on the same images the mean gain over zero-shot Florence-2 is 2.59 (95% bootstrap
  interval 2.00 to 3.13), and the fine-tuned caption scores higher on 68% of images. That is a
  real but modest improvement.
- **Training itself went as expected.** Validation loss fell from 3.53 to 0.73 and validation
  CIDEr rose from 16.9 to 291.3 (best at epoch 4 of 5), in 32 minutes on one H100.

Figures, per-epoch numbers, example captions and failure cases are in
`notebooks/02_results.ipynb`.

## Approach

| Question | Decision | Why |
|---|---|---|
| Which pretrained model? | `florence-community/Florence-2-base-ft` | Small enough to fine-tune fully on one GPU, 768 px input keeps collars and pockets visible, strong captioning pretraining, MIT licence. |
| Is fine-tuning needed? | Yes, and measured against zero-shot baselines | Zero-shot captioners write generic lines such as "A black tank top against a white background." They lack garment vocabulary and alt-text style. |
| Which data? | `tomytjandra/h-and-m-fashion-caption`, 20,491 image-caption pairs | Clean studio product photos with detailed descriptions. |
| Can its captions be used directly? | No, they are rewritten first | See below. |

### The captions had to be rewritten

`notebooks/01_data_exploration.ipynb` shows that none of the original captions works as alt-text:

- 76.7% are longer than the usual 125-character guideline (median 163 characters);
- none has punctuation or sentence case, so a screen reader reads them as one stream;
- 87.8% state something a camera cannot see (fibre content, linings, "soft", "recycled").

A captioner trained on them would learn to guess invisible facts. So each caption is rewritten
once by a text-only LLM (Qwen2.5-7B-Instruct) into one punctuated sentence that keeps the visible
facts and drops the rest. Every rewrite is checked automatically (length, punctuation, a word
list of non-visible terms, pattern mentioned, no back-of-garment details), sent back with the
reason up to twice if it fails, and excluded if it still fails.

Outcome: 19,363 of 20,491 rewrites (94.5%) passed. Median length 97 characters, 89.4% within 125.
The captioner is trained and evaluated only on rewrites that passed.

## Limitations

- **The training targets are silver labels.** The LLM rewrote text without seeing the image. In
  a hand read of 50 accepted rewrites I found one invented detail, three back-of-garment details
  that were reworded instead of removed, and two garbled phrases. Automatic counts over all
  accepted rewrites: about 2.4% still describe the back, and up to 0.3% mention side pockets
  that the source does not.
- **The model asserts small details it cannot always confirm.** Pockets, fly type and
  "detachable" belts are frequent in the training captions, and the model sometimes adds them
  when the photo does not settle it. CLIPScore is not sensitive to this, so no number here
  measures how often it happens. Checking a sample of captions against the photos by hand would
  be the next step.
- **Close-up photos.** Some catalogue images show only a detail. The model still describes a
  whole garment, including a length or sleeve type the crop does not show.
- **Multi-colour and printed garments.** Colour and pattern in the references come from catalogue
  labels, which are sometimes wrong for such items (a tie-dye t-shirt labelled "white"), and the
  model inherits that.
- **Outside H&M studio photos.** Tried on four public photos: a denim jacket on a chair and a red
  dress on a runway both got sensible garment-first captions with some wrong details (it placed
  a decoration on the front of a jacket photographed from the back). A painting produced a
  broken caption with a repeated word, and a photo of a bird was described as a bird. It is
  trained for product photos of single garments.
- **The non-visible-term check is a word list**, so "0% non-visible terms" means none of the
  listed words appear, not that every statement is visible.
- **Data use.** The images come from H&M catalogue data and are used here for coursework only.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `scripts/00_download.py` | Downloads the dataset and all models into `./.cache`. |
| 1 | `scripts/01_prepare_data.py` | Resizes images to 768 px on the short side and builds a 90/5/5 split, grouped so images sharing a description never straddle train and test. |
| 2 | `scripts/02_rewrite_captions.py` | Rewrites raw captions into alt-text and checks every rewrite. |
| 3 | `scripts/03_predict.py` | Captions the test split with any system (zero-shot baselines or the fine-tuned model). |
| 4 | `scripts/04_train.py` | Fine-tunes Florence-2 and keeps the checkpoint with the best validation CIDEr. |
| 5 | `scripts/05_evaluate.py` | Scores every system: BLEU-4, ROUGE-L, METEOR, CIDEr, CLIPScore, and alt-text checks. |
| | `scripts/caption.py` | Writes alt-text for your own images. |

The helper functions the scripts share (model loading, metrics, text checks, plotting) are in
`src/utils/`. The notebooks only read what the scripts produce.

Training settings: full fine-tune, AdamW, learning rate 1e-5 with 5% warm-up then linear decay,
batch size 16, 5 epochs, bf16 autocast, input = image + Florence-2's caption task prompt,
target = the alt-text.

The reported run used an earlier, longer version of `04_train.py` that could also stop and resume
across jobs. The current script is a shorter version with the same settings. Started from scratch,
it reproduced the logged losses of the reported run to within 0.002 over the first 600 steps
(for example 1.5729 against 1.5727 at step 500).

## How to run it

Everything stays inside this folder, including the virtual environment and all caches.

```bash
# once, on the login node
uv sync --python 3.12            # creates .venv from pyproject.toml / uv.lock
echo 'export SBATCH_ACCOUNT=<your account>' > env.local.sh   # the Slurm account to charge
source env.sh                    # activates .venv, points every cache at ./.cache
python scripts/00_download.py    # about 23 GB

# data preparation (CPU job), then the GPU work
sbatch slurm/prepare.sbatch
sbatch slurm/smoke.sbatch        # optional: confirms the GPU node can run the project
sbatch slurm/trial.sbatch        # optional: 300 trial rewrites to read by hand
sbatch slurm/pipeline.sbatch     # steps 2-5 in one job (needs a partition with a multi-hour limit)
```

The Slurm scripts are written for IU Quartz (H100 partitions). They take the account to charge
from `SBATCH_ACCOUNT`, which `env.sh` reads from `env.local.sh`. Submit from the project root, in
a shell where `env.sh` has been sourced.

**If the long queue is busy.** `h100-single` can have a wait of a day, while `h100-debug` starts
at once but stops jobs after one hour. Every step fits in an hour except the rewrite, which
picks up where it stopped, so the same work can be done in one-hour pieces. This is how the
results here were produced:

```bash
D="-p h100-debug --time=00:58:00"
sbatch $D slurm/rewrite.sbatch --batch-size 128        # step 2; resubmit until every caption is done
sbatch $D slurm/predict.sbatch                         # steps 3 + 5: zero-shot baselines, scores
sbatch slurm/train.sbatch --overfit 32 --epochs 60 --lr 5e-5   # optional sanity check, a few minutes
sbatch slurm/train.sbatch                              # step 4: training, about 35 minutes
sbatch $D slurm/predict.sbatch --systems florence_finetuned   # steps 3 + 5 for the fine-tuned model
```

`--dependency=afterok:<jobid>` chains them so they run unattended.

Caption your own image once training has finished:

```bash
source env.sh
python scripts/caption.py path/to/photo.jpg
```

To re-run a notebook: `jupyter nbconvert --to notebook --execute --inplace notebooks/02_results.ipynb`.
In this repository the notebook outputs that show H&M product photos have been cleared; the
charts and tables are kept. Re-running a notebook brings the photos back.

### The demo and the published model

The fine-tuned model and a demo page are on Hugging Face (links at the top of this file). The
demo is a small Gradio app that downloads the model and writes alt-text for an uploaded photo;
its code is under the **Files** tab of the Space. The files used to publish the model and the
demo are not part of this repository.

## Things that went wrong, and what fixed them

- **Weights turned to NaN on the first training step.** `transformers` 5 loads this checkpoint in
  float16, because that is how it is stored. In float16 Adam's epsilon (1e-8) underflows to zero
  and the first update divides 0 by 0. Loading with `dtype=torch.float32` fixes it. An overfit
  check on 32 examples (`04_train.py --overfit 32`) is what caught this before the long run.
- **Jobs took ten minutes to start.** The first read of a small file from Slate on a fresh compute
  node is slow, and importing `transformers` reads about 3,000 of them one at a time.
  `slurm/warm_cache.sh` reads them in parallel at the start of each job. On a fresh node this took
  about five minutes; on a node used before, a few seconds.
- **One rewrite job died with a GPU driver error** after 13,824 captions. It did not recur; the
  script resumed from where it stopped.

## What is where

```
notebooks/01_data_exploration.ipynb   the data, and why the captions needed rewriting
notebooks/02_results.ipynb            training curves, test results, examples, failure cases
outputs/results.md, results.csv       the results table
outputs/test_predictions.csv          every test image with each system's caption and CLIPScore
outputs/train_log.csv, val_log.csv    loss per 20 steps, validation numbers per epoch
outputs/rewrite_spotcheck.csv         100 random raw/rewritten pairs for reading by hand
outputs/rewrite_rejected.csv          the 1,128 rewrites that failed the checks, with reasons
outputs/checkpoints/best/             the fine-tuned model (not in git, 0.9 GB)
data/alt_text.jsonl                   every rewrite with its check result (not in git)
```

`data/`, `.cache/`, `.venv/` and the checkpoint are not in git. The pipeline re-creates them, and
the fine-tuned model can be downloaded from Hugging Face.

`.cache/` holds about 23 GB of downloads. The 15 GB Qwen model in it is only needed for step 2
and can be deleted once `data/alt_text.jsonl` exists.
