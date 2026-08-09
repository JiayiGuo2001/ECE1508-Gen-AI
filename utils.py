import json
from pathlib import Path
from collections import Counter
from random import Random

import h5py
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torch

from candidate_adapter import portable_image_path


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100, image_size=256, random_seed=123, compression='lzf'):
    """
    Create input files for training, validation, and testing.

    """

    supported_datasets = {'coco', 'flickr8k', 'flickr30k'}

    if dataset not in supported_datasets:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets are: {sorted(supported_datasets)}")

    if captions_per_image < 1:
        raise ValueError("captions_per_image must be at least 1.")

    if min_word_freq < 1:
        raise ValueError("min_word_freq must be at least 1.")

    if max_len < 1:
        raise ValueError("max_len must be at least 1.")

    if image_size < 1:
        raise ValueError("image_size must be at least 1.")

    if compression not in {'lzf', 'gzip', None}:
        raise ValueError("compression must be None, 'gzip', or 'lzf'.")

    karpathy_json_path = Path(karpathy_json_path).expanduser().resolve()
    print(f"Karpathy JSON path: {karpathy_json_path}")
    image_folder = Path(image_folder).expanduser().resolve()
    output_folder = Path(output_folder).expanduser().resolve()

    if not karpathy_json_path.is_file():
        raise FileNotFoundError(f"Karpathy JSON file not found: {karpathy_json_path}")

    if not image_folder.is_dir():
        raise NotADirectoryError(f"Image folder not found: {image_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    # Read Karpathy JSON
    with karpathy_json_path.open('r', encoding='utf-8') as j:
        data = json.load(j)

    if not isinstance(data, dict) or not isinstance(data.get('images'), list):
        raise ValueError("Invalid Karpathy JSON format. Expected a dictionary containing an 'images' list.")

    # Containers
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []

    word_freq = Counter()

    split_data = {
        'TRAIN': (train_image_paths, train_image_captions),
        'VAL': (val_image_paths, val_image_captions),
        'TEST': (test_image_paths, test_image_captions)
    }

    # Read image paths and captions
    for img in data['images']:
        source_split = img.get('split')

        if source_split in {'train', 'restval'}:
            split = 'TRAIN'
        elif source_split == 'val':
            split = 'VAL'
        elif source_split == 'test':
            split = 'TEST'
        else:
            continue

        captions = []

        for sentence in img.get('sentences', []):
            tokens = sentence.get('tokens', [])

            if not isinstance(tokens, list):
                continue

            if tokens and len(tokens) <= max_len:
                captions.append(tokens)
                if split == 'TRAIN':
                    word_freq.update(tokens)

        if len(captions) == 0:
            continue

        filename = img.get('filename')

        if not filename:
            raise ValueError("An image entry is missing 'filename'.")

        relative_path = Path(img.get('filepath', '')) / filename if dataset == 'coco' else Path(filename)
        path = image_folder / relative_path

        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        split_data[split][0].append(path)
        split_data[split][1].append(captions)

    # Sanity checks
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    print("\nDataset split summary:")
    print(f"TRAIN: {len(train_image_paths)} images")
    print(f"VAL:   {len(val_image_paths)} images")
    print(f"TEST:  {len(test_image_paths)} images")

    # Create word map from training captions only
    words = sorted(word for word, frequency in word_freq.items() if frequency >= min_word_freq)
    word_map = {word: index + 1 for index, word in enumerate(words)}

    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    print(f"Vocabulary size: {len(word_map)}")

    base_filename = f"{dataset}_{captions_per_image}_cap_per_img_{min_word_freq}_min_word_freq"

    # Save word map
    with (output_folder / f"WORDMAP_{base_filename}.json").open('w', encoding='utf-8') as j:
        json.dump(word_map, j, ensure_ascii=False)

    # Save metadata
    metadata = {
        'dataset': dataset,
        'captions_per_image': captions_per_image,
        'min_word_freq': min_word_freq,
        'max_len': max_len,
        'image_size': image_size,
        'compression': compression,
        'random_seed': random_seed,
        'split_image_counts': {
            'TRAIN': len(train_image_paths),
            'VAL': len(val_image_paths),
            'TEST': len(test_image_paths)
        },
        'vocabulary_size': len(word_map)
    }

    with (output_folder / f"METADATA_{base_filename}.json").open('w', encoding='utf-8') as j:
        json.dump(metadata, j, ensure_ascii=False, indent=2)

    rng = Random(random_seed)

    # Process TRAIN / VAL / TEST
    for split in ['TRAIN', 'VAL', 'TEST']:
        impaths, imcaps = split_data[split]

        print(f"\nProcessing {split}: {len(impaths)} images...")

        hdf5_path = output_folder / f"{split}_IMAGES_{base_filename}.hdf5"

        with h5py.File(str(hdf5_path), 'w') as h:
            h.attrs['captions_per_image'] = captions_per_image

            dataset_kwargs = {
                'shape': (len(impaths), 3, image_size, image_size),
                'dtype': 'uint8'
            }

            if len(impaths) > 0:
                dataset_kwargs['chunks'] = (1, 3, image_size, image_size)

                if compression is not None:
                    dataset_kwargs['compression'] = compression

            images = h.create_dataset('images', **dataset_kwargs)

            encoded_captions = []
            caption_lengths = []

            for i, path in enumerate(tqdm(impaths, desc=f"Processing {split}")):
                available_captions = imcaps[i]

                if len(available_captions) < captions_per_image:
                    captions = available_captions.copy()

                    while len(captions) < captions_per_image:
                        captions.append(rng.choice(available_captions))
                else:
                    captions = rng.sample(available_captions, k=captions_per_image)

                assert len(captions) == captions_per_image

                # Read image
                try:
                    with Image.open(path) as img:
                        img = img.convert('RGB')
                        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                        img = np.asarray(img, dtype=np.uint8)

                except (OSError, UnidentifiedImageError) as error:
                    raise ValueError(f"Could not read image {path}: {error}") from error

                if img.shape != (image_size, image_size, 3):
                    raise ValueError(f"Unexpected image shape for {path}: {img.shape}")

                if img.max() > 255:
                    raise ValueError(f"Image contains invalid pixel values: {path}")

                # HWC -> CHW
                img = img.transpose(2, 0, 1)

                assert img.shape == (3, image_size, image_size)

                # Save image
                images[i] = img

                # Encode captions
                for caption in captions:
                    encoded_caption = (
                        [word_map['<start>']]
                        + [word_map.get(word, word_map['<unk>']) for word in caption]
                        + [word_map['<end>']]
                        + [word_map['<pad>']] * (max_len - len(caption))
                    )

                    caption_length = len(caption) + 2
                    encoded_captions.append(encoded_caption)
                    caption_lengths.append(caption_length)

        # Sanity checks
        expected_caption_count = len(impaths) * captions_per_image

        assert len(encoded_captions) == expected_caption_count
        assert len(caption_lengths) == expected_caption_count

        if encoded_captions:
            assert all(len(caption) == max_len + 2 for caption in encoded_captions)

        # Save captions
        with (output_folder / f"{split}_CAPTIONS_{base_filename}.json").open('w', encoding='utf-8') as j:
            json.dump(encoded_captions, j)

        # Save caption lengths
        with (output_folder / f"{split}_CAPLENS_{base_filename}.json").open('w', encoding='utf-8') as j:
            json.dump(caption_lengths, j)
        
        # Preserve the original image identity and location for downstream
        # caption reranking. HDF5 image arrays alone do not retain filenames.
        image_records = [
            {
                'image_id': path.name,
                'image_path': portable_image_path(path),
            }
            for path in impaths
        ]
        with (output_folder / f"{split}_IMAGE_PATHS_{base_filename}.json").open(
            'w', encoding='utf-8'
        ) as j:
            json.dump(image_records, j, ensure_ascii=False, indent=2)


        print(f"{split} complete: {len(impaths)} images, {len(encoded_captions)} captions")

    print(f"\nFinished processing {dataset}.")
    print(f"Output folder: {output_folder}")

def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best, encoder_name=None):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    :param encoder_name: pretrained encoder architecture used by this run
    """
    if encoder_name is None:
        encoder_name = getattr(encoder, 'encoder_name', 'resnet101')

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder_name': encoder_name,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = f'checkpoint_{data_name}_{encoder_name}.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


        
        
