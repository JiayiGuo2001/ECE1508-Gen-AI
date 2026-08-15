# Attention-Based Image Captioning with CLIP Reranking

This repository implements an image-captioning pipeline built around a
pretrained CNN encoder, an LSTM decoder with attention. The
decoder can generate an n-best candidate set with deterministic beam search or
temperature-controlled stochastic beam sampling. A CLIP-based reranker then
selects the candidate that best matches the input image.


## Model architecture

- ImageNet-pretrained encoders:
  - `resnet101`
  - `resnext101_32x8d`
  - `wide_resnet50_2`
  - `convnext_tiny`
  - `efficientnet_b3` (`EfficientNet-B3` is also accepted)

- LSTM decoder with soft attention

When encoder fine-tuning is enabled, the code freezes the complete backbone
first and then trains feature-container children from index 5 onward. For the
ResNet-family encoders, this corresponds to `layer2`, `layer3`, and `layer4`.


## Installation

Create and activate an environment, then
install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The evaluation package uses Java for the PTB tokenizer, METEOR, and SPICE.
Java 11 is recommended; SPICE may fail with newer Java releases.

Verify the metric and CLIP environment with:

```bash
python3 -m rerank.check_env
```

The first use of a torchvision encoder, CLIP, BLIP, or some evaluation metrics
may download pretrained weights or supporting files.

## 1. Prepare a dataset

The preprocessor expects a Karpathy-format JSON file such as
`dataset_coco.json` or `dataset_flickr30k.json` and the corresponding image
directory.

The data is suggested to put under foler data/coco or data/flickr30k

Example for Flickr30k:

```bash
python3 create_input_files.py \
  --dataset flickr30k \
  --karpathy-json data/flickr30k/dataset_flickr30k.json \
  --image-folder data/flickr30k/images \
  --output-folder data/flickr30k/processed \
  --captions-per-image 5 \
  --min-word-freq 3 \
  --max-len 50 \
  --image-size 256 \
  --seed 123
```

Example for COCO, where the Karpathy JSON contains paths such as `train2014`
and `val2014`:

```bash
python3 create_input_files.py \
  --dataset coco \
  --karpathy-json data/coco2014/dataset_coco.json \
  --image-folder data/coco2014/images \
  --output-folder data/coco2014/processed \
  --captions-per-image 5 \
  --min-word-freq 3 \
  --max-len 50
```

## 2. Configure training

Set `data_folder`, `data_name`, and the model settings in `hyperparams.yaml`:

```yaml
data_folder: data/flickr30k/processed
data_name: flickr30k_5_cap_per_img_3_min_word_freq

emb_dim: 512
attention_dim: 512
decoder_dim: 512
dropout: 0.5
encoder_name: resnet101

epochs: 15
batch_size: 64
workers: 4
encoder_lr: 0.0001
decoder_lr: 0.0004
grad_clip: 5.0
alpha_c: 1.0
print_freq: 100

fine_tune_encoder: false
checkpoint: null

sampling_temperature: 0.0
sampling_seed: 123
```

`data_name` must exactly match the suffix generated during preprocessing. A
checkpoint must use the same dataset word map and the same encoder architecture
as its configuration.

## 3. Train the captioning model

Start a new training run with `checkpoint: null`:

```bash
python3 train.py --config hyperparams.yaml
```

Training automatically uses CUDA when it is available. The decoder is always
trained. With `fine_tune_encoder: false`, the encoder is frozen and kept in
evaluation mode.

Each epoch writes the latest checkpoint in the working directory:

```text
checkpoint_<data_name>_<encoder_name>.pth.tar
```

When validation BLEU-4 improves, it also writes:

```text
BEST_checkpoint_<data_name>_<encoder_name>.pth.tar
```

### Fine-tune from a frozen-encoder checkpoint

A common two-stage procedure is:

1. Train with `fine_tune_encoder: false` and `checkpoint: null`.
2. Point `checkpoint` to the saved best checkpoint.
3. Set `fine_tune_encoder: true` and use the smaller `encoder_lr`.
4. Set `epochs` to a value greater than the checkpoint's next epoch.

For example:

```yaml
fine_tune_encoder: true
encoder_lr: 0.00005
checkpoint: no_fine_tune_flicker30k/BEST_checkpoint_flickr30k_5_cap_per_img_3_min_word_freq_resnet101.pth.tar
epochs: 30
```

Then run the same training command:

```bash
python3 train.py --config hyperparams.yaml
```

## 4. Generate caption candidates

Use the best checkpoint for inference and reranking:

```bash
python3 generate_captions.py \
  --config hyperparams.yaml \
  --checkpoint no_fine_tune_flicker30k/BEST_checkpoint_flickr30k_5_cap_per_img_3_min_word_freq_resnet101.pth.tar \
  --beam-size 5 \
  --max-steps 50 \
  --temperature 0 \
  --num-images 0 \
  --candidates-out beams/flickr30k_beams.jsonl
```

Temperature has the following behavior:

- `0`: deterministic beam search.
- `0 < T < 1`: sharper distribution and more conservative samples.
- `T > 1`: flatter distribution and more diverse samples.


## 6. Beams files
All beams*.jsonl files are saved under beams/ folder

## 6. Run CLIP reranking and evaluation

Compare beam top-1, random selection, pure CLIP, decoder-CLIP fusion, and the
candidate-pool oracle:

```bash
python3 -m rerank.run_experiment \
  --candidates beams/flickr30k_beams.jsonl \
  --captions data/flickr30k/evaluation/captions.txt \
  --alphas 0,0.25,0.5,0.75,1 \
  --metrics bleu,meteor,rouge,cider \
  --cache cache/flickr30k_clip_img_emb.npz \
  --out results/flickr30k_selectors.csv
```

For fusion weight `alpha`, the selector computes:

```text
score = alpha * zscore(CLIP cosine similarity)
      + (1 - alpha) * zscore(length-normalized decoder logprob)
```

Therefore, `alpha=1` is pure CLIP and `alpha=0` is the length-normalized
decoder score. The oracle uses the references and represents the approximate
selection ceiling of the existing candidate pool; it is not a deployable
selector.

Use `--fake-clip` only to test pipeline plumbing without loading CLIP:

```bash
python3 -m rerank.run_experiment \
  --candidates beams/flickr30k_beams.jsonl \
  --captions data/flickr30k/evaluation/captions.txt \
  --fake-clip \
  --out results/plumbing_test.csv
```

Fake CLIP scores are deterministic random values and are not valid experiment
results.

To include SPICE in the final evaluation, add it explicitly:

```bash
--metrics bleu,meteor,rouge,cider,spice
```

## Optional BLIP baseline

Generate BLIP candidates for the same images as an existing candidate file:

```bash
python3 -m baselines.blip_captioner \
  --like beams/flickr30k_beams.jsonl \
  --out beams/flickr30k_blip.jsonl \
  --beam-size 5 \
  --limit 0
```

Then add the baseline to the selector experiment:

```bash
python3 -m rerank.run_experiment \
  --candidates beams/flickr30k_beams.jsonl \
  --captions data/flickr30k/evaluation/captions.txt \
  --compare beams/flickr30k_blip.jsonl \
  --compare-name blip \
  --out results/flickr30k_with_blip.csv
```

BLIP requires the Hugging Face `transformers` package and downloads its
checkpoint on first use.

## Common problems

####  Checkpoints

Trained model checkpoints are not included in this repository due to their large file size. If you require the checkpoints, please reach out to us.

#### Reference Implementation: 
The data preprocessing and training pipeline in this project is adapted from the PyTorch image captioning tutorial by sgrvinod:
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning