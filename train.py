import argparse
import json
import os
import time
import yaml

import torch
import torchvision.transforms as transforms

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

from models import Encoder, DecoderWithAttention
from datasets import CaptionDataset
from utils import (
    AverageMeter,
    accuracy,
    adjust_learning_rate,
    clip_gradient,
    save_checkpoint,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Default hyperparameters
data_folder = "data"
data_name = ""

emb_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0.5
encoder_name = "resnet101"

start_epoch = 0
epochs = 120
epochs_since_improvement = 0

batch_size = 32
workers = 1

encoder_lr = 1e-4
decoder_lr = 4e-4

grad_clip = 5.0
alpha_c = 1.0

best_bleu4 = 0.0
print_freq = 100

fine_tune_encoder = False
checkpoint = None


# Configuration
def load_hyperparams(config_path):
    """Load hyperparameters from a YAML file."""

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Hyperparameter file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def apply_hyperparams(config):
    """Override default hyperparameters with values from YAML."""

    global data_folder, data_name
    global emb_dim, attention_dim, decoder_dim, dropout, encoder_name
    global start_epoch, epochs, epochs_since_improvement
    global batch_size, workers
    global encoder_lr, decoder_lr
    global grad_clip, alpha_c
    global best_bleu4, print_freq
    global fine_tune_encoder, checkpoint

    data_folder = config.get("data_folder", data_folder)
    data_name = config.get("data_name", data_name)

    emb_dim = config.get("emb_dim", emb_dim)
    attention_dim = config.get("attention_dim", attention_dim)
    decoder_dim = config.get("decoder_dim", decoder_dim)
    dropout = config.get("dropout", dropout)
    encoder_name = config.get("encoder_name", encoder_name)

    start_epoch = config.get("start_epoch", start_epoch)
    epochs = config.get("epochs", epochs)
    epochs_since_improvement = config.get(
        "epochs_since_improvement",
        epochs_since_improvement,
    )

    batch_size = config.get("batch_size", batch_size)
    workers = config.get("workers", workers)

    encoder_lr = config.get("encoder_lr", encoder_lr)
    decoder_lr = config.get("decoder_lr", decoder_lr)

    grad_clip = config.get("grad_clip", grad_clip)
    alpha_c = config.get("alpha_c", alpha_c)

    best_bleu4 = config.get("best_bleu4", best_bleu4)
    print_freq = config.get("print_freq", print_freq)

    fine_tune_encoder = config.get(
        "fine_tune_encoder",
        fine_tune_encoder,
    )
    checkpoint = config.get("checkpoint", checkpoint)


def main(config_path="hyperparams.yaml"):
    """Train and validate the image captioning model."""

    global best_bleu4
    global epochs_since_improvement
    global start_epoch

    # Load configuration ONCE
    config = load_hyperparams(config_path)
    apply_hyperparams(config)

    word_map_file = os.path.join(
        data_folder,
        f"WORDMAP_{data_name}.json",
    )

    with open(word_map_file, "r", encoding="utf-8") as f:
        word_map = json.load(f)

    if checkpoint is None:

        encoder = Encoder(encoder_name=encoder_name)
        encoder.fine_tune(fine_tune_encoder)

        decoder = DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=emb_dim,
            decoder_dim=decoder_dim,
            vocab_size=len(word_map),
            encoder_dim=encoder.encoder_dim,
            dropout=dropout,
        )

        decoder_optimizer = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad,
                decoder.parameters(),
            ),
            lr=decoder_lr,
        )

        if fine_tune_encoder:
            encoder_optimizer = torch.optim.Adam(
                filter(
                    lambda p: p.requires_grad,
                    encoder.parameters(),
                ),
                lr=encoder_lr,
            )
        else:
            encoder_optimizer = None

    else:

        checkpoint_data = torch.load(
            checkpoint,
            map_location=device,
            weights_only=False,
        )

        start_epoch = checkpoint_data["epoch"] + 1
        epochs_since_improvement = checkpoint_data[
            "epochs_since_improvement"
        ]
        best_bleu4 = checkpoint_data["bleu-4"]

        decoder = checkpoint_data["decoder"]
        decoder_optimizer = checkpoint_data["decoder_optimizer"]

        encoder = checkpoint_data["encoder"]
        encoder_optimizer = checkpoint_data["encoder_optimizer"]

        requested_encoder_name = Encoder.normalize_name(encoder_name)
        checkpoint_encoder_name = checkpoint_data.get(
            "encoder_name",
            getattr(encoder, "encoder_name", "resnet101"),
        )
        checkpoint_encoder_name = Encoder.normalize_name(
            checkpoint_encoder_name
        )

        if checkpoint_encoder_name != requested_encoder_name:
            raise ValueError(
                "Checkpoint encoder mismatch: "
                f"hyperparams.yaml requests '{requested_encoder_name}', "
                f"but the checkpoint uses '{checkpoint_encoder_name}'."
            )

        # Supply metadata missing from checkpoints made before encoder
        # selection was configurable.
        encoder.encoder_name = checkpoint_encoder_name
        if not hasattr(encoder, "encoder_dim"):
            encoder.encoder_dim = decoder.encoder_dim

        # Fine-tune encoder if requested but checkpoint did not
        # previously have an encoder optimizer.
        if fine_tune_encoder and encoder_optimizer is None:

            encoder.fine_tune(True)

            encoder_optimizer = torch.optim.Adam(
                filter(
                    lambda p: p.requires_grad,
                    encoder.parameters(),
                ),
                lr=encoder_lr,
            )

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # Data normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform = transforms.Compose([normalize])

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, "TRAIN", transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, "VAL", transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    for epoch in range(start_epoch, epochs):

        # Stop after 20 epochs without BLEU improvement
        if epochs_since_improvement >= 20:
            print("Stopping early: no improvement for 20 epochs.")
            break

        # Reduce learning rate every 8 epochs without improvement
        if (
            epochs_since_improvement > 0
            and epochs_since_improvement % 8 == 0
        ):
            adjust_learning_rate(decoder_optimizer, 0.8)

            if encoder_optimizer is not None:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train(
            train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
        )

        recent_bleu4 = validate(
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            word_map=word_map,
        )
        is_best = recent_bleu4 > best_bleu4

        if is_best:
            best_bleu4 = recent_bleu4
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

            print(
                f"\nEpochs since last improvement: "
                f"{epochs_since_improvement}\n"
            )

        save_checkpoint(
            data_name,
            epoch,
            epochs_since_improvement,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            recent_bleu4,
            is_best,
            encoder_name=encoder.encoder_name,
        )


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    decoder.train()

    # Important:
    # If the encoder is frozen, keep BatchNorm etc. in evaluation mode.
    if encoder_optimizer is None:
        encoder.eval()
    else:
        encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):

        data_time.update(time.time() - start)

        imgs = imgs.to(device, non_blocking=True)
        caps = caps.to(device, non_blocking=True)
        caplens = caplens.to(device, non_blocking=True)

        imgs = encoder(imgs)

        scores, caps_sorted, decode_lengths, alphas, _ = decoder(imgs, caps, caplens)

        # Remove <start> from targets
        targets = caps_sorted[:, 1:]

        # .data contains only the valid, non-padded values.
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=True).data

        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=True).data

        loss = criterion(scores, targets)

        # Doubly stochastic attention regularization
        attention_reg = (
            (1.0 - alphas.sum(dim=1)) ** 2
        ).mean()

        loss = loss + alpha_c * attention_reg

        decoder_optimizer.zero_grad()

        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        loss.backward()

        # Gradient clipping
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)

            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()

        if encoder_optimizer is not None:
            encoder_optimizer.step()

        num_words = sum(decode_lengths)

        top5 = accuracy(
            scores,
            targets,
            5,
        )

        losses.update(
            loss.item(),
            num_words,
        )

        top5accs.update(
            top5,
            num_words,
        )

        batch_time.update(
            time.time() - start
        )

        start = time.time()

        if i % print_freq == 0:

            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                f"Batch Time {batch_time.val:.3f} "
                f"({batch_time.avg:.3f})\t"
                f"Data Load Time {data_time.val:.3f} "
                f"({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} "
                f"({losses.avg:.4f})\t"
                f"Top-5 Accuracy {top5accs.val:.3f} "
                f"({top5accs.avg:.3f})"
            )


def validate(val_loader, encoder, decoder, criterion, word_map):
    decoder.eval()
    encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    references = []
    hypotheses = []

    start = time.time()

    with torch.no_grad():

        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            imgs = imgs.to(device, non_blocking=True)
            caps = caps.to(device, non_blocking=True)
            caplens = caplens.to(device, non_blocking=True)

            imgs = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]

            # Keep unpacked scores for caption predictions
            scores_copy = scores.clone()

            # Remove padding
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=True).data

            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=True).data

            loss = criterion(scores, targets)

            attention_reg = (
                (1.0 - alphas.sum(dim=1)) ** 2
            ).mean()

            loss = loss + alpha_c * attention_reg

            num_words = sum(decode_lengths)

            losses.update(
                loss.item(),
                num_words,
            )

            top5 = accuracy(
                scores,
                targets,
                5,
            )

            top5accs.update(
                top5,
                num_words,
            )

            batch_time.update(
                time.time() - start
            )

            start = time.time()

            if i % print_freq == 0:

                print(
                    f"Validation: [{i}/{len(val_loader)}]\t"
                    f"Batch Time {batch_time.val:.3f} "
                    f"({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} "
                    f"({losses.avg:.4f})\t"
                    f"Top-5 Accuracy {top5accs.val:.3f} "
                    f"({top5accs.avg:.3f})"
                )

            # Decoder sorted the batch by caption length,
            # so all reference captions need the same ordering.
            allcaps = allcaps[sort_ind.cpu()]

            for j in range(allcaps.size(0)):

                img_caps = allcaps[j].tolist()

                img_captions = [
                    [
                        word
                        for word in caption
                        if word
                        not in {
                            word_map["<start>"],
                            word_map["<pad>"],
                        }
                    ]
                    for caption in img_caps
                ]

                references.append(img_captions)

            preds = scores_copy.argmax(dim=2)
            preds = preds.tolist()

            for j, pred in enumerate(preds):

                hypothesis = pred[:decode_lengths[j]]

                hypotheses.append(hypothesis)

            assert len(references) == len(hypotheses)

    bleu4 = corpus_bleu(
        references,
        hypotheses,
    )

    print(
        f"\n"
        f" * LOSS - {losses.avg:.3f}, "
        f"TOP-5 ACCURACY - {top5accs.avg:.3f}, "
        f"BLEU-4 - {bleu4:.4f}\n"
    )

    return bleu4


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train the image captioning model"
    )

    parser.add_argument(
        "--config",
        "-c",
        default="hyperparams.yaml",
        help="Path to hyperparameters YAML file",
    )

    args = parser.parse_args()

    main(config_path=args.config)
