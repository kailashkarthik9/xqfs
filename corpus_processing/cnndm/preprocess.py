import argparse
import glob
import hashlib
import json
import logging
import random
from enum import Enum
from os import path

import nltk.data
from multiprocess.pool import Pool
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.DEBUG)

SENTENCE_BOUNDARIES = {
    '.',
    '?',
    '!'
}
SENTENCE_BOUNDARY_EXCEPTIONS = {
    ')',
    "'",
    '"'
}
DATASET_SPLITS = [
    'test',
    'valid',
    'train',
]


class PreProcessingStage(Enum):
    SENTENCE_SPLIT = 'sentence_split'
    TVT_SPLIT = 'tvt_split'
    TRANSLATION_PREP = 'translation_prep'

    def __str__(self):
        return self.value


class CnnDmPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def perform_sentence_split(raw_stories_dir, sentence_ized_stories_dir):
        logger.info('Sentence segmentation stage started')
        logger.info('Loading NLTK sentence tokenization model')
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        logger.info('Performing sentence segmentation on CNN/DM stories')
        for file in tqdm(glob.iglob(path.join(raw_stories_dir, '*'))):
            article_lines = list()
            summary_sentences = list()
            has_summary_started = False
            with open(file) as fp:
                story_lines = fp.read().split('\n')
            for line in story_lines:
                line = line.strip()
                if line == '':
                    continue
                if line == '@highlight':
                    has_summary_started = True
                    continue
                if has_summary_started:
                    summary_sentences.append(line)
                else:
                    article_lines.append(line)
            article_text = ''
            for line in article_lines:
                if (line[-1] in SENTENCE_BOUNDARIES) or (
                        len(line) > 1 and
                        line[-1] in SENTENCE_BOUNDARY_EXCEPTIONS and line[:-1].strip()[-1] in SENTENCE_BOUNDARIES):
                    article_text += line + ' '
                else:
                    article_text += line + '. '
            article_sentences = sent_detector.tokenize(article_text.strip())
            output_json = {
                'article': article_sentences,
                'summary': summary_sentences
            }
            file_name = file.split('/')[-1]
            json.dump(output_json, open(path.join(sentence_ized_stories_dir, file_name + '.json'), 'w'), indent=4)
        logger.info('Sentence segmentation stage completed!')

    @staticmethod
    def hashhex(s):
        """Returns a heximal formated SHA1 hash of the input string."""
        h = hashlib.sha1()
        h.update(s.encode('utf-8'))
        return h.hexdigest()

    @staticmethod
    def perform_tvt_split(sentence_ized_stories_dir, tvt_dir, urls_dir):
        logger.info('TVT segmentation stage started')
        logger.info('Loading TVT URL maps')
        corpus_split_stories = {}
        for split in tqdm(DATASET_SPLITS):
            stories = []
            for line in open(path.join(urls_dir, 'mapping_' + split + '.txt')):
                stories.append(CnnDmPreprocessor.hashhex(line.strip()))
            corpus_split_stories[split] = {key.strip() for key in stories}

        logger.info('Partitioning stories into TVT sets')
        train_files, valid_files, test_files = [], [], []
        for file in tqdm(glob.iglob(path.join(sentence_ized_stories_dir, '*.json'))):
            file_name = file.split('/')[-1].split('.')[0]
            if file_name in corpus_split_stories['valid']:
                valid_files.append(file)
            elif file_name in corpus_split_stories['test']:
                test_files.append(file)
            elif file_name in corpus_split_stories['train']:
                train_files.append(file)
            else:
                raise Exception('File found that is not a part of any corpus split!')

        def _format_to_lines(file_):
            json_data = json.load(open(file_))
            json_data['id'] = file_.split('/')[-1].split('.')[0]
            return json_data

        logger.info('Aggregating TVT json files')
        corpus_splits = {
            'train': train_files,
            'valid': valid_files,
            'test': test_files
        }
        for split in tqdm(DATASET_SPLITS):
            pool = Pool(args.n_cpus)
            dataset = []
            for d in pool.imap_unordered(_format_to_lines, corpus_splits[split]):
                dataset.append(d)
            pool.close()
            pool.join()
            json.dump(dataset, open(path.join(tvt_dir, split + '.json'), 'w'), indent=4)
        logger.info('TVT segmentation stage completed!')

    @staticmethod
    def perform_translation_prep(tvt_dir, translation_files_dir):
        logger.info('Translation Prep stage started')
        for split in DATASET_SPLITS:
            logger.info(f'Processing {split} split...')
            data_file = path.join(tvt_dir, split + '.json')
            combined_file = path.join(translation_files_dir, split + '.merged.txt')
            combined_map_file = path.join(translation_files_dir, split + '.merged.map.json')
            sorted_file = path.join(translation_files_dir, split + '.sorted.txt')
            sorted_map_file = path.join(translation_files_dir, split + '.sorted.map.json')

            logger.info('Merging all articles in split')
            split_data = json.load(open(data_file))
            combined_data = list()
            combined_data_map = list()
            for idx, article in tqdm(enumerate(split_data)):
                combined_data_map.append({
                    'idx': idx,
                    'id': article['id'],
                    'start': len(combined_data),
                    'end': len(combined_data) + len(article['article'])
                })
                combined_data.extend(article['article'])
            json.dump(combined_data_map, open(combined_map_file, 'w'), indent=4)
            with open(combined_file, 'w') as fp:
                fp.write('\n'.join(combined_data))

            logger.info('Sorting the merged file')
            combined_data_map = json.load(open(combined_map_file))
            with open(combined_file, 'r') as fp:
                combined_data = fp.read().split('\n')

            logger.info('Test 1 - all sentences are present')
            assert combined_data_map[-1]['end'] == len(combined_data)

            sorted_data = sorted(enumerate(combined_data), key=lambda i: len(i[1].split()), reverse=True)
            sorted_combined_data = [i[1] for i in sorted_data]
            sorted_combined_data_indices = {i[0]: idx for idx, i in enumerate(sorted_data)}

            json.dump(sorted_combined_data_indices, open(sorted_map_file, 'w'))
            with open(sorted_file, 'w') as fp:
                fp.write('\n'.join(sorted_combined_data))

            sorted_combined_data_indices = json.load(open(sorted_map_file, 'r'))
            with open(sorted_file, 'r') as fp:
                sorted_combined_data = fp.read().split('\n')

            logger.info('Test 2 - all sentences are present in sorted data and map')
            assert len(sorted_combined_data) == len(combined_data)
            assert len(sorted_combined_data_indices) == len(combined_data)

            logger.info('Test 3 - Random sample 100 docs and verify sorted index mapping')
            for _ in tqdm(range(100)):
                doc = random.choice(combined_data_map)
                article = split_data[doc['idx']]['article']
                article_from_sorted_file = list()
                article_from_merged_file = list()
                for sent_idx in range(doc['start'], doc['end']):
                    article_from_sorted_file.append(
                        sorted_combined_data[sorted_combined_data_indices[str(sent_idx)]].strip())
                    article_from_merged_file.append(combined_data[sent_idx].strip())
                assert article == article_from_sorted_file
                assert article == article_from_merged_file
            logger.info(f'All tests OK. {split} split prepared for translation')
        logger.info('Translation Prep stage completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-raw_stories_dir", type=str, default='data/cnndm/raw_stories')
    parser.add_argument("-sentence_ized_stories_dir", type=str, default='data/cnndm/sentence_ized_stories')
    parser.add_argument("-tvt_dir", type=str, default='data/cnndm/tvt')
    parser.add_argument("-urls_dir", type=str, default='metadata/urls')
    parser.add_argument("-translation_files_dir", type=str, default='data/cnndm/translation')
    parser.add_argument("-n_cpus", type=int, default=8)
    parser.add_argument('-stage', required=True, type=PreProcessingStage, choices=list(PreProcessingStage))
    args = parser.parse_args()

    if args.stage == PreProcessingStage.SENTENCE_SPLIT:
        CnnDmPreprocessor.perform_sentence_split(args.raw_stories_dir, args.sentence_ized_stories_dir)
    elif args.stage == PreProcessingStage.TVT_SPLIT:
        CnnDmPreprocessor.perform_tvt_split(args.sentence_ized_stories_dir, args.tvt_dir, args.urls_dir)
    elif args.stage == PreProcessingStage.TRANSLATION_PREP:
        CnnDmPreprocessor.perform_translation_prep(args.tvt_dir, args.translation_files_dir)
