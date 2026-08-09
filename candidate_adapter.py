"""Adapter between decoder-native beams and the reranker JSONL contract."""

import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_image_path(image_path):
    """Resolve a candidate path, treating relative paths as repo-relative."""
    image_path = Path(image_path).expanduser()
    if image_path.is_absolute():
        return image_path.resolve()
    return (PROJECT_ROOT / image_path).resolve()


def portable_image_path(image_path):
    """Use a repo-relative path when the image is stored in this project."""
    resolved_path = resolve_image_path(image_path)
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        # Images outside the repository cannot be represented by a valid
        # repository-relative path, so preserve their absolute location.
        return str(resolved_path)


def tokens_to_text(token_ids, word_map):
    """Detokenize one decoder sequence and remove model control tokens."""
    reverse_word_map = {index: word for word, index in word_map.items()}
    words = []

    for token_id in token_ids:
        word = reverse_word_map.get(int(token_id), "<unk>")
        if word == "<end>":
            break
        if word not in {"<start>", "<pad>"}:
            words.append(word)

    return " ".join(words)


def beams_to_candidates(beams, word_map):
    """Convert decoder beams to reranker candidate dictionaries.

    Decoder beams contain token IDs and raw cumulative natural-log
    probabilities. The reranker receives detokenized text and the same raw
    scores; no length normalization is applied here.
    """
    ordered_beams = sorted(
        beams,
        key=lambda beam: float(beam["logprob"]),
        reverse=True,
    )

    candidates = []
    seen_text = set()
    for beam_index, beam in enumerate(ordered_beams):
        if "token_ids" not in beam or "logprob" not in beam:
            raise KeyError(
                f"Decoder beam {beam_index} needs token_ids and logprob."
            )

        logprob = float(beam["logprob"])
        if not math.isfinite(logprob):
            raise ValueError(
                f"Decoder beam {beam_index} has a non-finite logprob."
            )

        text = tokens_to_text(beam["token_ids"], word_map).strip()
        normalized_text = text.casefold()
        if not text or normalized_text in seen_text:
            continue

        seen_text.add(normalized_text)
        candidates.append({"text": text, "logprob": logprob})

    if not candidates:
        raise ValueError("Decoder produced no non-empty, unique candidates.")

    return candidates


def make_candidate_record(image_record, beams, word_map):
    """Attach image identity to adapted beams using the reranker schema."""
    image_id = image_record.get("image_id")
    image_path = image_record.get("image_path")
    if not image_id or not image_path:
        raise ValueError("image_record needs image_id and image_path.")

    return {
        "image_id": str(image_id),
        "image_path": portable_image_path(image_path),
        "candidates": beams_to_candidates(beams, word_map),
    }


def save_candidate_records(records, output_path):
    """Write one reranker candidate record per JSONL line."""
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def relativize_candidate_file(input_path, output_path):
    """Rewrite candidate image paths using portable repository-relative paths."""
    input_path = Path(input_path).expanduser()
    records = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = record.get("image_path")
            if not image_path:
                raise ValueError(
                    f"Candidate record on line {line_number} has no image_path."
                )
            record["image_path"] = portable_image_path(image_path)
            records.append(record)

    save_candidate_records(records, output_path)
    return len(records)


def build_test_image_records(
    karpathy_json_path,
    images_dir,
    dataset_name,
    max_caption_length,
):
    """Reconstruct TEST image identities in the preprocessing order."""
    karpathy_json_path = Path(karpathy_json_path).expanduser()
    images_dir = Path(images_dir).expanduser()

    with karpathy_json_path.open("r", encoding="utf-8") as source_file:
        source = json.load(source_file)

    records = []
    for image_record in source.get("images", []):
        if image_record.get("split") != "test":
            continue

        valid_caption_exists = any(
            isinstance(sentence.get("tokens"), list)
            and sentence["tokens"]
            and len(sentence["tokens"]) <= max_caption_length
            for sentence in image_record.get("sentences", [])
        )
        if not valid_caption_exists:
            continue

        filename = image_record.get("filename")
        if not filename:
            raise ValueError("A TEST image entry is missing 'filename'.")

        if dataset_name == "coco":
            relative_path = Path(image_record.get("filepath", "")) / filename
        else:
            relative_path = Path(filename)

        records.append(
            {
                "image_id": filename,
                "image_path": portable_image_path(images_dir / relative_path),
            }
        )

    return records


def load_test_image_records(
    data_folder,
    data_name,
    image_manifest_path=None,
    karpathy_json_path=None,
    images_dir=None,
):
    """Load image IDs/paths required by the reranker candidate format."""
    data_folder = Path(data_folder)
    inferred_manifest = data_folder / f"TEST_IMAGE_PATHS_{data_name}.json"
    manifest_path = (
        Path(image_manifest_path).expanduser()
        if image_manifest_path
        else inferred_manifest
    )

    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            raw_records = json.load(manifest_file)
    elif image_manifest_path:
        raise FileNotFoundError(f"Image manifest not found: {manifest_path}")
    elif karpathy_json_path and images_dir:
        metadata_path = data_folder / f"METADATA_{data_name}.json"
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        raw_records = build_test_image_records(
            karpathy_json_path=karpathy_json_path,
            images_dir=images_dir,
            dataset_name=metadata["dataset"],
            max_caption_length=metadata["max_len"],
        )
    else:
        raise FileNotFoundError(
            f"Image manifest not found: {inferred_manifest}. "
            "For an existing processed dataset, provide both "
            "--karpathy-json and --images-dir."
        )

    records = []
    seen_ids = set()
    for index, record in enumerate(raw_records):
        if isinstance(record, str):
            image_path = record
            image_id = Path(record).name
        elif isinstance(record, dict):
            image_path = record.get("image_path")
            image_id = record.get("image_id")
            if not image_id and image_path:
                image_id = Path(image_path).name
        else:
            raise TypeError(
                f"Invalid image manifest entry at index {index}: {record!r}"
            )

        if not image_id or not image_path:
            raise ValueError(
                f"Image manifest entry {index} needs image_id and image_path."
            )
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image_id in manifest: {image_id}")

        seen_ids.add(image_id)
        records.append(
            {
                "image_id": str(image_id),
                "image_path": portable_image_path(image_path),
            }
        )

    missing_paths = [
        record["image_path"]
        for record in records
        if not resolve_image_path(record["image_path"]).is_file()
    ]
    if missing_paths:
        examples = ", ".join(missing_paths[:3])
        raise FileNotFoundError(
            f"{len(missing_paths)} TEST image paths do not exist. "
            f"Examples: {examples}"
        )

    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert candidate image paths to repo-relative paths."
    )
    parser.add_argument("--input", required=True, help="source candidate JSONL")
    parser.add_argument("--output", required=True, help="converted JSONL")
    arguments = parser.parse_args()

    record_count = relativize_candidate_file(
        arguments.input,
        arguments.output,
    )
    print(f"Converted {record_count} records to {arguments.output}")
