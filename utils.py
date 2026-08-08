import os
import json
from pathlib import Path
from collections import Counter
from random import Random
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import torch


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100, image_size=256, random_seed=123, compression='lzf'):
    """
    Create input files for the dataset.
    Reference: a=PYTorch-TUT
    """
    supported_datasets = ['coco', 'flickr8k', 'flickr30k']

    if dataset not in supported_datasets:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets are: {supported_datasets}")
    if captions_per_image < 1:
        raise ValueError("captions_per_image must be at least 1.")
    if min_word_freq < 1:
        raise ValueError("min_word_freq must be at least 1.")
    if max_len < 1 or image_size < 1:
        raise ValueError("max_len and image_size must be at least 1.")
    if compression not in ['lzf', 'gzip', None]:
        raise ValueError("compression must be None, 'gzip', or 'lzf'")

    karapthy_json_path = Path(karpathy_json_path).expanduser().resolve()
    image_folder = Path(image_folder).expanduser().resolve()
    output_folder = Path(output_folder).expanduser().resolve()

    if not karapthy_json_path.is_file():
        raise FileNotFoundError(f"Karpathy JSON file not found: {karapthy_json_path}")
    if not image_folder.is_dir():
        raise NotADirectoryError(f"Image folder not found: {image_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Read the Karpathy JSON
    with karapthy_json_path.open('r', encoding='utf-8') as j:
        data = json.load(j)
    if not isinstance(data.get('images'), list):
        raise ValueError("Invalid Karpathy JSON format. Expected a dictionary with an 'images' key.")

    # Read images and captions
    train_image_paths = []
    train_captions = []
    val_image_paths = []
    val_captions = []
    test_image_paths = []
    test_captions = []
    word_freq = Counter()

    split_data ={
        'TRAIN': (train_image_paths, train_captions),
        'VAL': (val_image_paths, val_captions),
        'TEST': (test_image_paths, test_captions)
    }

    for img in data['images']:
        source_split = img.get('split')
        split = 'TRAIN' if source_split in {'train', 'restval'} else source_split.upper() if source_split in {
            'val', 'test'} else None
        if split is None:
            continue

        captions = []
        for c in img.get('sentences', []):
            tokens = c.get('tokens', [])
            if tokens and len(tokens) <= max_len:
                captions.append(' '.join(tokens))
                if split == 'TRAIN':
                    word_freq.update(tokens)

        if len(captions) == 0:
            continue

        relative_path = Path(img.get('filepath', '')) / img['filename'] if dataset == 'coco' else Path(img['filename'])
        path = image_folder / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        split_data[split][0].append(path)
        split_data[split][1].append(captions)

        # Create word map
        words = sorted(w for w, frequency in word_freq.items() if frequency >= min_word_freq)
        word_map = {k: v + 1 for v, k in enumerate(words)}
        word_map['<unk>'] = len(word_map) + 1
        word_map['<start>'] = len(word_map) + 1
        word_map['<end>'] = len(word_map) + 1
        word_map['<pad>'] = 0

        base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

        with (output_folder / ('WORDMAP_' + base_filename + '.json')).open('w', encoding='utf-8') as j:
            json.dump(word_map, j, ensure_ascii=False)

        metadata = {
            'dataset': dataset,
            'captions_per_image': captions_per_image,
            'min_word_freq': min_word_freq,
            'max_len': max_len,
            'image_size': image_size,
            'compression': compression,
            'random_seed': random_seed,
            'split_image_counts': {split: len(paths) for split, (paths, _) in split_data.items()},
            'vocabulary_size': len(word_map)
        }

        with (output_folder / ('METADATA_' + base_filename + '.json')).open('w', encoding='utf-8') as j:
            json.dump(metadata, j)

        # save images to HDF5 files, captions to JSON files, caption lengths to JSON files
        rng = Random(random_seed)
        for impaths, imcaps, split in [(train_image_paths, train_captions, 'TRAIN'),
                        (val_image_paths, val_captions, 'VAL'),
                        (test_image_paths, test_captions, 'TEST')]:
            hdf5_path = output_folder / (split + '_IMAGES_' + base_filename + '.hdf5')
            with h5py.File(str(hdf5_path), 'w') as h:
                h.attrs['captions_per_image'] = captions_per_image

                images = h.create_dataset(
                    'images', 
                    (len(impaths), 3, image_size, image_size), 
                    dtype='uint8', 
                    chunks=(1, 3, image_size, image_size) if impaths else None,
                    compression=compression if impaths else None,
                )

                print(f"Processing {split} images and captions...")

                encoded_captions = []
                caption_lengths = []

                for i, path in enumerate(tqdm(impaths)):
                    if len(imcaps[i]) < captions_per_image:
                        captions = imcaps[i] + [rng.choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                    else:
                        captions = rng.sample(imcaps[i], k=captions_per_image)

                    assert len(captions) == captions_per_image

                    # Read images
                    try:
                        with Image.open(path) as img:
                            img = img.convert('RGB')
                            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                            img = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
                    except (OSError, UnidentifiedImageError) as error:
                        raise ValueError('Could not read image %s: %s' % (path, os.errorror)) from os.error

                    images[i] = img

                    for j, c in enumerate(captions):
                        encoded_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    caption_lengths.append(len(c) + 2)  # +2 for <start> and <end>
                    encoded_captions.append(encoded_c)

                    assert images.shape[0] * captions_per_image == len(encoded_captions) == len(caption_lengths)

                    with (output_folder / (split + '_CAPTIONS_' + base_filename + '.json')).open('w', encoding='utf-8') as j:
                        json.dump(encoded_captions, j)

                    with (output_folder / (split + '_CAPLENS_' + base_filename + '.json')).open('w', encoding='utf-8') as j:
                        json.dump(caption_lengths, j)


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


        
        
