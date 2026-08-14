"""Generate captions for unique images in the processed TEST split."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml

from candidate_adapter import (
    beams_to_candidates,
    load_test_image_records,
    make_candidate_record,
    save_candidate_records,
    tokens_to_text,
)
from datasets import CaptionDataset
from utils import report_model_size


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sample_expansions(log_scores, num_samples, generator):
    """Sample indices without replacement from unnormalized log scores.

    Gumbel-top-k implements sampling without replacement while avoiding the
    numerical underflow that can occur when exponentiating long-sequence beam
    scores. 
    """
    if log_scores.dim() != 1:
        raise ValueError("Sampling expects a one-dimensional score tensor.")
    if num_samples < 1 or num_samples > log_scores.numel():
        raise ValueError("Invalid number of beam expansions to sample.")

    cpu_scores = log_scores.detach().double().cpu()
    uniform = torch.rand(
        cpu_scores.shape,
        generator=generator,
        dtype=cpu_scores.dtype,
    )
    tiny = torch.finfo(uniform.dtype).tiny
    uniform = uniform.clamp_(min=tiny, max=1.0 - torch.finfo(uniform.dtype).eps)
    gumbel_noise = -torch.log(-torch.log(uniform))
    sampled_indices = (cpu_scores + gumbel_noise).topk(num_samples).indices
    return sampled_indices.to(log_scores.device)


def resolve_device(device_name):
    """Resolve ``auto`` to the best device available in this environment."""
    if device_name != "auto":
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_checkpoint(checkpoint_path, device):
    """Load the trained encoder and decoder stored by ``train.py``."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with PyTorch versions without ``weights_only``.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if "encoder" not in checkpoint or "decoder" not in checkpoint:
        raise KeyError(
            "Checkpoint must contain both 'encoder' and 'decoder' models."
        )

    encoder = checkpoint["encoder"].to(device)
    decoder = checkpoint["decoder"].to(device)
    encoder.eval()
    decoder.eval()

    encoder_name = checkpoint.get(
        "encoder_name",
        getattr(encoder, "encoder_name", "resnet101"),
    )

    return encoder, decoder, encoder_name


@torch.no_grad()
def generate_caption_beams(
    encoder,
    decoder,
    image,
    word_map,
    beam_size=5,
    max_steps=50,
    temperature=0.0,
    sampling_seed=None,
):
    """Return beam sequences and cumulative natural-log probabilities.

    :param encoder: trained image encoder
    :param decoder: trained attention decoder
    :param image: normalized tensor shaped ``(3, H, W)`` or ``(1, 3, H, W)``
    :param word_map: mapping from vocabulary tokens to integer IDs
    :param beam_size: number of active beam-search candidates
    :param max_steps: maximum number of generated tokens
    :param temperature: 0 uses deterministic beam search; a positive value
        enables stochastic beam sampling. Values below 1 are more conservative,
        while values above 1 increase diversity.
    :param sampling_seed: optional seed for reproducible stochastic beams
    :return: beam records containing ``token_ids`` and raw ``logprob``
    """
    temperature = float(temperature)
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite, non-negative number.")

    stochastic = temperature > 0
    sampling_generator = None
    if stochastic:
        sampling_generator = torch.Generator(device="cpu")
        if sampling_seed is None:
            sampling_generator.seed()
        else:
            sampling_generator.manual_seed(int(sampling_seed))

    required_tokens = {"<start>", "<end>", "<pad>"}
    missing_tokens = required_tokens.difference(word_map)
    if missing_tokens:
        raise KeyError(
            f"Word map is missing required tokens: {sorted(missing_tokens)}"
        )

    device = next(encoder.parameters()).device
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.size(0) != 1:
        raise ValueError("Beam search expects exactly one image.")

    image = image.to(device)
    encoder_out = encoder(image)
    encoder_dim = encoder_out.size(-1)
    encoder_out = encoder_out.reshape(1, -1, encoder_dim)

    vocabulary_size = decoder.vocab_size
    if beam_size > vocabulary_size:
        raise ValueError(
            f"beam_size ({beam_size}) cannot exceed vocabulary size "
            f"({vocabulary_size})."
        )

    current_beam_size = beam_size
    encoder_out = encoder_out.expand(current_beam_size, -1, -1)

    previous_words = torch.full(
        (current_beam_size, 1),
        word_map["<start>"],
        dtype=torch.long,
        device=device,
    )
    sequences = previous_words
    sequence_scores = torch.zeros(
        current_beam_size,
        1,
        device=device,
    )
    sampling_sequence_scores = torch.zeros_like(sequence_scores)

    completed_sequences = []
    completed_scores = []

    hidden, cell = decoder.init_hidden_state(encoder_out)

    for step in range(1, max_steps + 1):
        embeddings = decoder.embedding(previous_words).squeeze(1)
        attention_encoding, _ = decoder.attention(encoder_out, hidden)
        gate = decoder.sigmoid(decoder.f_beta(hidden))
        attention_encoding = gate * attention_encoding

        hidden, cell = decoder.decode_step(
            torch.cat([embeddings, attention_encoding], dim=1),
            (hidden, cell),
        )

        logits = decoder.fc(hidden)
        raw_token_scores = F.log_softmax(logits, dim=1)
        raw_scores = raw_token_scores + sequence_scores.expand_as(
            raw_token_scores
        )

        if stochastic:
            sampling_token_scores = F.log_softmax(
                logits / temperature,
                dim=1,
            )
            sampling_scores = (
                sampling_token_scores
                + sampling_sequence_scores.expand_as(sampling_token_scores)
            )
        else:
            sampling_scores = raw_scores

        # At the first step every beam is identical, so expand only the
        # first one. At later steps, expand every active candidate.
        if step == 1:
            candidate_raw_scores = raw_scores[0]
            candidate_sampling_scores = sampling_scores[0]
        else:
            candidate_raw_scores = raw_scores.reshape(-1)
            candidate_sampling_scores = sampling_scores.reshape(-1)

        if stochastic:
            top_words = sample_expansions(
                candidate_sampling_scores,
                current_beam_size,
                sampling_generator,
            )
            top_scores = candidate_raw_scores[top_words]
            selected_sampling_scores = candidate_sampling_scores[top_words]
        else:
            top_scores, top_words = candidate_raw_scores.topk(
                current_beam_size
            )
            selected_sampling_scores = top_scores

        previous_sequence_indices = torch.div(
            top_words,
            vocabulary_size,
            rounding_mode="floor",
        )
        next_word_indices = top_words % vocabulary_size

        sequences = torch.cat(
            [
                sequences[previous_sequence_indices],
                next_word_indices.unsqueeze(1),
            ],
            dim=1,
        )

        is_complete = next_word_indices == word_map["<end>"]
        complete_indices = is_complete.nonzero(as_tuple=False).squeeze(1)
        incomplete_indices = (~is_complete).nonzero(as_tuple=False).squeeze(1)

        if complete_indices.numel() > 0:
            completed_sequences.extend(
                sequences[complete_indices].tolist()
            )
            completed_scores.extend(top_scores[complete_indices].tolist())

        current_beam_size -= complete_indices.numel()
        if current_beam_size == 0:
            break

        sequences = sequences[incomplete_indices]
        previous_beam_indices = previous_sequence_indices[
            incomplete_indices
        ]
        hidden = hidden[previous_beam_indices]
        cell = cell[previous_beam_indices]
        encoder_out = encoder_out[previous_beam_indices]
        sequence_scores = top_scores[incomplete_indices].unsqueeze(1)
        sampling_sequence_scores = selected_sampling_scores[
            incomplete_indices
        ].unsqueeze(1)
        previous_words = next_word_indices[incomplete_indices].unsqueeze(1)

    scored_sequences = list(zip(completed_sequences, completed_scores))

    # If max_steps was reached before every active beam emitted <end>, retain
    # those beams too instead of losing potentially useful candidates.
    if current_beam_size > 0:
        active_scores = sequence_scores.squeeze(1).tolist()
        scored_sequences.extend(zip(sequences.tolist(), active_scores))

    scored_sequences.sort(key=lambda item: item[1], reverse=True)

    beams = []
    for token_ids, logprob in scored_sequences:
        beams.append(
            {
                "logprob": float(logprob),
                "token_ids": token_ids,
            }
        )

    if not beams:
        raise RuntimeError(
            "Beam search did not produce any candidates."
        )

    return beams


def caption_image_candidates(
    encoder,
    decoder,
    image,
    word_map,
    beam_size=5,
    max_steps=50,
    temperature=0.0,
    sampling_seed=None,
):
    """Return the full n-best list required by the CLIP reranker."""
    beams = generate_caption_beams(
        encoder=encoder,
        decoder=decoder,
        image=image,
        word_map=word_map,
        beam_size=beam_size,
        max_steps=max_steps,
        temperature=temperature,
        sampling_seed=sampling_seed,
    )
    return beams_to_candidates(beams, word_map)


def caption_image(
    encoder,
    decoder,
    image,
    word_map,
    beam_size=5,
    max_steps=50,
    temperature=0.0,
    sampling_seed=None,
):
    """Generate the highest-log-probability caption (legacy top-1 API)."""
    beams = generate_caption_beams(
        encoder=encoder,
        decoder=decoder,
        image=image,
        word_map=word_map,
        beam_size=beam_size,
        max_steps=max_steps,
        temperature=temperature,
        sampling_seed=sampling_seed,
    )
    for beam in beams:
        text = tokens_to_text(beam["token_ids"], word_map).strip()
        if text:
            return text, beam["token_ids"]
    raise RuntimeError("Beam search produced no non-empty caption.")


def generate_test_captions(
    checkpoint_path,
    data_folder,
    data_name,
    word_map_path=None,
    image_manifest_path=None,
    karpathy_json_path=None,
    images_dir=None,
    beam_size=5,
    max_steps=50,
    start_index=0,
    num_images=10,
    device_name="auto",
    temperature=0.0,
    sampling_seed=None,
):
    """Caption unique images from the processed TEST split."""
    temperature = float(temperature)
    if start_index < 0:
        raise ValueError("start_index cannot be negative.")
    if num_images < 0:
        raise ValueError("num_images cannot be negative; use 0 for all.")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite, non-negative number.")

    device = resolve_device(device_name)
    checkpoint_path = Path(checkpoint_path).expanduser()
    data_folder = Path(data_folder).expanduser()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_folder.is_dir():
        raise NotADirectoryError(f"Data folder not found: {data_folder}")

    if word_map_path is None:
        word_map_path = data_folder / f"WORDMAP_{data_name}.json"
    else:
        word_map_path = Path(word_map_path).expanduser()

    with word_map_path.open("r", encoding="utf-8") as word_map_file:
        word_map = json.load(word_map_file)

    encoder, decoder, encoder_name = load_checkpoint(
        checkpoint_path,
        device,
    )
    report_model_size(
        encoder,
        decoder,
        checkpoint_path=checkpoint_path,
    )

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    test_dataset = CaptionDataset(
        str(data_folder),
        data_name,
        "TEST",
        transform=normalize,
    )

    total_images = len(test_dataset) // test_dataset.cpi
    image_records = load_test_image_records(
        data_folder=data_folder,
        data_name=data_name,
        image_manifest_path=image_manifest_path,
        karpathy_json_path=karpathy_json_path,
        images_dir=images_dir,
    )
    if len(image_records) != total_images:
        test_dataset.h.close()
        raise ValueError(
            "TEST image manifest is not aligned with the processed HDF5: "
            f"manifest has {len(image_records)} images, HDF5 has "
            f"{total_images}."
        )

    if start_index >= total_images:
        test_dataset.h.close()
        raise IndexError(
            f"start_index {start_index} is outside a TEST split with "
            f"{total_images} images."
        )

    stop_index = total_images
    if num_images > 0:
        stop_index = min(start_index + num_images, total_images)

    print(f"Device: {device}")
    print(f"Encoder: {encoder_name}")
    print(f"Checkpoint: {checkpoint_path}")
    if temperature > 0:
        print(
            f"Decoding: stochastic beam sampling "
            f"(temperature={temperature:g}, seed={sampling_seed})"
        )
    else:
        print("Decoding: deterministic beam search")
    print(f"TEST images: {total_images}\n")

    results = []
    try:
        for image_index in range(start_index, stop_index):
            # CaptionDataset repeats each image once per reference caption.
            dataset_index = image_index * test_dataset.cpi
            image, _, _, reference_tokens = test_dataset[dataset_index]

            beams = generate_caption_beams(
                encoder=encoder,
                decoder=decoder,
                image=image,
                word_map=word_map,
                beam_size=beam_size,
                max_steps=max_steps,
                temperature=temperature,
                sampling_seed=(
                    None
                    if sampling_seed is None
                    else int(sampling_seed) + image_index
                ),
            )

            references = []
            for tokens in reference_tokens.tolist():
                reference = tokens_to_text(tokens, word_map)
                if reference not in references:
                    references.append(reference)

            image_record = image_records[image_index]
            result = make_candidate_record(
                image_record=image_record,
                beams=beams,
                word_map=word_map,
            )
            candidates = result["candidates"]
            results.append(result)

            print(f"Image {image_index}: {image_record['image_id']}")
            print("Candidates:")
            for candidate_index, candidate in enumerate(candidates, start=1):
                print(
                    f"  {candidate_index}. [{candidate['logprob']:.4f}] "
                    f"{candidate['text']}"
                )
            print("References:")
            for reference in references:
                print(f"  - {reference}")
            print()
    finally:
        test_dataset.h.close()

    return results


def load_config(config_path):
    """Load paths and dataset names from the training YAML configuration."""
    config_path = Path(config_path).expanduser()
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate captions for images in the TEST split."
    )
    parser.add_argument(
        "--config",
        "-c",
        default="hyperparams.yaml",
        help="training YAML containing data_folder and data_name",
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "trained checkpoint; defaults to config checkpoint or the "
            "BEST checkpoint matching data_name and encoder_name"
        ),
    )
    parser.add_argument(
        "--word-map",
        help="optional WORDMAP JSON path; inferred from the YAML by default",
    )
    parser.add_argument(
        "--image-manifest",
        help=(
            "TEST image ID/path JSON; inferred from the processed data "
            "folder when available"
        ),
    )
    parser.add_argument(
        "--karpathy-json",
        help="original Karpathy split JSON, used when no image manifest exists",
    )
    parser.add_argument(
        "--images-dir",
        help="original image directory, used with --karpathy-json",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "stochastic beam-sampling temperature; 0 keeps deterministic "
            "beam search, <1 is conservative, and >1 is more diverse"
        ),
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="optional reproducibility seed for stochastic beam sampling",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="number of unique TEST images to caption; use 0 for all",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        "--candidates-out",
        dest="output",
        default="data/beams.jsonl",
        help="reranker-compatible candidate JSONL output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    data_folder = config.get("data_folder")
    data_name = config.get("data_name")
    encoder_name = str(config.get("encoder_name", "resnet101"))
    encoder_name = encoder_name.strip().lower().replace("-", "_")

    if not data_folder or not data_name:
        raise ValueError(
            "The YAML config must define both data_folder and data_name."
        )

    checkpoint_path = args.checkpoint or config.get("checkpoint")
    if not checkpoint_path:
        checkpoint_path = (
            f"BEST_checkpoint_{data_name}_{encoder_name}.pth.tar"
        )

    temperature = args.temperature
    if temperature is None:
        temperature = float(config.get("sampling_temperature", 0.0))
    sampling_seed = args.sampling_seed
    if sampling_seed is None:
        sampling_seed = config.get("sampling_seed")

    results = generate_test_captions(
        checkpoint_path=checkpoint_path,
        data_folder=data_folder,
        data_name=data_name,
        word_map_path=args.word_map,
        image_manifest_path=args.image_manifest,
        karpathy_json_path=args.karpathy_json,
        images_dir=args.images_dir,
        beam_size=args.beam_size,
        max_steps=args.max_steps,
        start_index=args.start_index,
        num_images=args.num_images,
        device_name=args.device,
        temperature=temperature,
        sampling_seed=sampling_seed,
    )

    output_path = Path(args.output).expanduser()
    save_candidate_records(results, output_path)
    print(f"Saved {len(results)} candidate lists to {output_path}")


if __name__ == "__main__":
    main()
