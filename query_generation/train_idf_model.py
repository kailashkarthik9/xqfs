import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-translation_files_dir", type=str, default='data/cnndm/translation')
    parser.add_argument("-sampling_split", type=str, default='valid')
    parser.add_argument("-sampling_count", type=int, default=50)
    args = parser.parse_args()
