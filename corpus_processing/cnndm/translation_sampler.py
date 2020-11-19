import argparse
from os import path


def sample_sentences():
    with open(path.join(args.translation_files_dir, args.sampling_split + '.sorted.txt')) as fp:
        sample = fp.read().split('\n')[:args.sampling_count]
    with open(path.join(args.translation_files_dir,
                        args.sampling_split + '.' + str(args.sampling_count) + '.sorted.txt'), 'w') as fp:
        fp.write('\n'.join(sample))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-translation_files_dir", type=str, default='data/cnndm/translation')
    parser.add_argument("-sampling_split", type=str, default='valid')
    parser.add_argument("-sampling_count", type=int, default=50)
    args = parser.parse_args()
    sample_sentences()
