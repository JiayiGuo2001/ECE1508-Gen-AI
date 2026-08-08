import argparse

from utils import create_input_files

def parse_args():
    parser = argparse.ArgumentParser(description="Create input files for the dataset.")
    parser.add_argument('--dataset', required=True, choices=('coco', 'flickr8k', 'flickr30k'))
    parser.add_argument('--karpathy-json', required=True, help='path to dataset_*.json from the Karpathy splits')
    parser.add_argument('--image-folder', required=True, help='folder containing the source images')
    parser.add_argument('--output-folder', required=True, help='folder in which processed files will be written')
    parser.add_argument('--captions-per-image', type=int, default=5)
    parser.add_argument('--min-word-freq', type=int, default=5)
    parser.add_argument('--max-len', type=int, default=50)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--compression', choices=('lzf', 'gzip', 'none'), default='lzf')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    create_input_files(
        dataset=args.dataset,
        karpathy_json_path=args.karpathy_json,
        image_folder=args.image_folder,
        captions_per_image=args.captions_per_image,
        min_word_freq=args.min_word_freq,
        output_folder=args.output_folder,
        max_len=args.max_len,
        image_size=args.image_size,
        random_seed=args.seed,
        compression=None if args.compression == 'none' else args.compression
    )