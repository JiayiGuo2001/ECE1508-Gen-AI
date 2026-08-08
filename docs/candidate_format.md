# Candidate file format

This is the only interface between the captioning model and the reranker. Hand
this page to whoever owns the decoder.

## What the reranker needs

A JSONL file — one JSON object per line, one line per image:

```json
{"image_id": "1000268201_693b08cb0e.jpg",
 "image_path": "data/Flicker8k_Dataset/1000268201_693b08cb0e.jpg",
 "candidates": [
   {"text": "a child in a pink dress is climbing stairs", "logprob": -8.21},
   {"text": "a little girl climbing into a wooden playhouse", "logprob": -9.03},
   {"text": "a girl in a pink dress on the stairs", "logprob": -9.44}
 ]}
```

| Field | Required | Notes |
|---|---|---|
| `image_id` | yes | Must match the filename used in the captions file, so references can be looked up. |
| `image_path` | yes | Path CLIP will open. Relative to repo root is fine. |
| `candidates[].text` | yes | Detokenized caption string. No `<start>`/`<end>` tokens. |
| `candidates[].logprob` | yes | **Cumulative** log-probability of the whole sequence (natural log, negative). The fusion selector needs this. |

Order doesn't matter — the reranker takes an explicit argmax rather than trusting
position — but beam order (best first) is conventional and easier to eyeball.

## The one change needed in the decoder

Beam search already computes all of this internally. The usual implementation
throws it away and returns only the best beam. Return the **full n-best list
with each beam's score** instead:

```python
# before
return best_beam.tokens

# after
return [{"text": detokenize(b.tokens), "logprob": float(b.score)} for b in beams]
```

If beam scores are already length-normalized on your side, say so — the reranker
normalizes by token count itself and would otherwise do it twice.

## Please: diverse candidates

Standard beam search on a small LSTM tends to return five near-identical strings
differing by an article. When that happens, reranking has nothing to select
between and the measured gain is zero no matter how good the reranker is.

Diverse beam search or top-k / nucleus sampling costs almost nothing to switch
on and makes the whole reranking experiment meaningful. If the decoder supports
a diversity penalty, please expose it as a flag.

## Verify before handing it over

```bash
python -m rerank.candidates --check data/beams.jsonl --images data/Flicker8k_Dataset/
```

Checks for missing fields, duplicate ids, empty captions, missing log-probs,
images that don't exist, and candidate lists where every entry is identical.
Exit code 0 means the file is usable.

## Testing before the model is ready

The reranker can fabricate realistic candidate lists from ground-truth
captions, so it does not block on the decoder:

```bash
python -m rerank.candidates --captions data/Flickr8k.token.txt \
    --images data/Flicker8k_Dataset/ --out data/mock_candidates.jsonl
```
