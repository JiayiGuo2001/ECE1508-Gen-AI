"""Generate captions for unique images in the processed TEST split."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml

from dataset import CaptionDataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
def caption_image(encoder, decoder, image, word_map, beam_size=5, max_steps=50):
    """Generate one caption from a normalized image using beam search.

    :param encoder: trained image encoder
    :param decoder: trained attention decoder
    :param image: normalized tensor shaped ``(3, H, W)`` or ``(1, 3, H, W)``
    :param word_map: mapping from vocabulary tokens to integer IDs
    :param beam_size: number of active beam-search candidates
    :param max_steps: maximum number of generated tokens
    :return: ``(caption_text, token_ids)``
    """
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

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
        raise ValueError("caption_image expects exactly one image.")

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

        scores = F.log_softmax(decoder.fc(hidden), dim=1)
        scores = scores + sequence_scores.expand_as(scores)

        # At the first step every beam is identical, so expand only the
        # first one. At later steps, expand every active candidate.
        if step == 1:
            top_scores, top_words = scores[0].topk(current_beam_size)
        else:
            top_scores, top_words = scores.reshape(-1).topk(
                current_beam_size
            )

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
        previous_words = next_word_indices[incomplete_indices].unsqueeze(1)

    if completed_sequences:
        best_index = max(
            range(len(completed_scores)),
            key=completed_scores.__getitem__,
        )
        token_ids = completed_sequences[best_index]
    else:
        # No beam emitted <end> before max_steps; return the best active one.
        best_index = sequence_scores.squeeze(1).argmax().item()
        token_ids = sequences[best_index].tolist()

    caption = tokens_to_text(token_ids, word_map)
    return caption, token_ids


def tokens_to_text(token_ids, word_map):
    """Convert encoded tokens to readable text and remove control tokens."""
    reverse_word_map = {index: word for word, index in word_map.items()}
    words = []

    for token_id in token_ids:
        word = reverse_word_map.get(int(token_id), "<unk>")
        if word == "<end>":
            break
        if word not in {"<start>", "<pad>"}:
            words.append(word)

    return " ".join(words)


def generate_test_captions(
    checkpoint_path,
    data_folder,
    data_name,
    word_map_path=None,
    beam_size=5,
    max_steps=50,
    start_index=0,
    num_images=10,
    device_name="auto",
):
    """Caption unique images from the processed TEST split."""
    if start_index < 0:
        raise ValueError("start_index cannot be negative.")
    if num_images < 0:
        raise ValueError("num_images cannot be negative; use 0 for all.")

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

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    test_dataset = CaptionDataset(
        str(data_folder),
        data_name,
        "TEST",
        transform=normalize,
    )

    total_images = len(test_dataset) // test_dataset.cpi
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
    print(f"TEST images: {total_images}\n")

    results = []
    try:
        for image_index in range(start_index, stop_index):
            # CaptionDataset repeats each image once per reference caption.
            dataset_index = image_index * test_dataset.cpi
            image, _, _, reference_tokens = test_dataset[dataset_index]

            generated_caption, generated_token_ids = caption_image(
                encoder=encoder,
                decoder=decoder,
                image=image,
                word_map=word_map,
                beam_size=beam_size,
                max_steps=max_steps,
            )

            references = []
            for tokens in reference_tokens.tolist():
                reference = tokens_to_text(tokens, word_map)
                if reference not in references:
                    references.append(reference)

            result = {
                "image_index": image_index,
                "caption": generated_caption,
                "token_ids": generated_token_ids,
                "references": references,
                "encoder_name": encoder_name,
            }
            results.append(result)

            print(f"Image {image_index}")
            print(f"Generated: {generated_caption}")
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
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
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
        help="optional JSON file to receive captions and references",
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

    results = generate_test_captions(
        checkpoint_path=checkpoint_path,
        data_folder=data_folder,
        data_name=data_name,
        word_map_path=args.word_map,
        beam_size=args.beam_size,
        max_steps=args.max_steps,
        start_index=args.start_index,
        num_images=args.num_images,
        device_name=args.device,
    )

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, ensure_ascii=False)
        print(f"Saved {len(results)} captions to {output_path}")


if __name__ == "__main__":
    main()
