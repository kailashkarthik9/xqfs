import argparse
import glob
import logging
import pickle
import random
from enum import Enum

from query_generation.keyphrase_extraction.launch import load_local_embedding_distributor, extract_keyphrases
from query_generation.keyphrase_extraction.postagging import PosTaggingSpacy
import json
from os import path
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.DEBUG)

DATASET_SPLITS = [
    'test',
    'valid',
    'train',
]

# Keyphrases extracted per document
KEYPHRASES_EXTRACTED_PER_DOCUMENT = 10
# Max queries per document is a hyper-parameter
MAX_QUERIES_PER_DOCUMENT = 3
# This threshold is chosen by inspecting the cumulative distribution created in the statistics plotting phase
KEYPHRASE_SCORE_THRESHOLD = 0.7
# This separator must be shared by Pegasus' tokenizer so that it is not split
QUERY_SEPARATOR = '[Q]'
# The size of the validation set
VAL_SET_SIZE = 20000


class QfsCorpusGenerationStage(Enum):
    KEYPHRASE_EXTRACTION = 'keyphrase_extraction'
    STATISTICS_PLOTTING = 'statistics_plotting'
    QFS_CORPUS_CREATION = 'qfs_corpus_creation'
    TRANSFORMER_FORMATTING = 'transformer_formatting'
    HYPOTHESIS_TESTER_CORPUS_CREATION = 'hypothesis_tester_corpus_creation'
    FULL_CORPUS_CREATION = 'full_corpus_creation'


class QfsCorpusGenerator:
    def __init__(self):
        self.embedding_distributor = None
        self.pos_tagger = None
        self.idf_model = None

    @staticmethod
    def get_qfs(keyphrase, summary_sentences):
        qfs = list()
        for sentence in summary_sentences:
            if keyphrase in sentence:
                qfs.append(sentence)
        return qfs

    def perform_keyphrase_extraction(self, tvt_dir, keyphrase_dir, sent2vec_model, idf_model, splits_to_exclude):
        logger.info('Keyphrase extraction stage started')
        if self.embedding_distributor is None:
            logger.info('Loading sent2vec model')
            self.embedding_distributor = load_local_embedding_distributor(sent2vec_model)
        if self.pos_tagger is None:
            logger.info('Loading POS tagger')
            self.pos_tagger = PosTaggingSpacy()
        if self.idf_model is None:
            logger.info('Loading IDF model')
            self.idf_model = pickle.load(open(idf_model, 'rb'))
        for split in DATASET_SPLITS:
            if split in splits_to_exclude:
                logger.info(f'Skipping {split} split')
                continue
            logger.info(f'Extracting keyphrases for {split} split')
            split_data = json.load(open(path.join(tvt_dir, split + '.json')))
            for data in tqdm(split_data):
                summary_text = ' '.join(data['summary'])
                keyphrases, scores, synsets = extract_keyphrases(self.embedding_distributor, self.pos_tagger,
                                                                 self.idf_model,
                                                                 summary_text, KEYPHRASES_EXTRACTED_PER_DOCUMENT, 'en')
                keyphrase_json = [
                    {
                        'keyphrase': keyphrase,
                        'summary': self.get_qfs(keyphrase, data['summary']),
                        'score': score,
                        'synset': synset,
                    }
                    for keyphrase, score, synset in zip(keyphrases, scores, synsets)
                ]
                data['keyphrases'] = sorted(keyphrase_json, key=lambda kp: kp['score'], reverse=True)
            json.dump(split_data, open(path.join(keyphrase_dir, split + '.query.json'), 'w'), indent=4)
        logger.info('Keyphrase extraction stage completed!')

    @staticmethod
    def perform_statistics_plotting(keyphrase_dir):
        logger.info('Statistics plotting stage started')
        doc_keyphrases = list()
        keyphrase_lengths = list()
        keyphrase_scores = list()
        keyphrase_cumulative_scores = [0 for _ in range(10)]

        for dataset_split_file in glob.iglob(path.join(keyphrase_dir, '*.json')):
            split = dataset_split_file.split('/')[-1].split('.')[0]
            logger.info(f'Extracting statistics for split {split}')
            if split in splits_to_exclude:
                logger.info(f'Skipping {split} split')
                continue
            dataset_split = json.load(open(dataset_split_file))
            for doc in dataset_split:
                keyphrases = doc['keyphrases']
                # Add statistics
                doc_keyphrases.append(len(keyphrases))
                for keyphrase in keyphrases:
                    keyphrase_scores.append(keyphrase['score'])
                    keyphrase_lengths.append(len(keyphrase['summary']))
                    for idx in range(10):
                        if keyphrase['score'] > idx / 10:
                            keyphrase_cumulative_scores[idx] += 1

        logger.info('Generating statistic plots')
        logger.info(f'Documents : {len(doc_keyphrases)}, Keyphrases : {len(keyphrase_scores)}')

        plt.figure()
        sns.histplot(doc_keyphrases, fill=True, bins=range(0, 11, 1))
        plt.xlabel('Number of Keyphrases')
        plt.ylabel('Documents')
        plt.title('Distribution of Keyphrase Count per Document')
        plt.savefig(path.join(keyphrase_dir, 'keyphrase_counts.png'))

        plt.figure()
        sns.histplot(keyphrase_lengths, fill=True, bins=range(0, 6, 1))
        plt.xlabel('Length')
        plt.ylabel('Keyphrases')
        plt.title('Distribution of Keyphrase Lengths')
        plt.savefig(path.join(keyphrase_dir, 'keyphrase_lengths.png'))

        plt.figure()
        sns.histplot(keyphrase_scores, fill=True, bins=[i / 10 for i in range(0, 11, 1)])
        plt.xlabel('Score')
        plt.ylabel('Keyphrases')
        plt.title('Distribution of Keyphrase Scores')
        plt.savefig(path.join(keyphrase_dir, 'keyphrase_scores.png'))

        plt.figure()
        sns.barplot(y=keyphrase_cumulative_scores, x=[i / 10 for i in range(10)], color='dodgerblue')
        plt.xlabel('Score Greater Than')
        plt.ylabel('Keyphrases')
        plt.title('Cumulative Distribution of Keyphrase Scores')
        plt.savefig(path.join(keyphrase_dir, 'keyphrase_cumulative_scores.png'))

        logger.info('Statistics plotting stage completed!')

    @staticmethod
    def perform_qfs_corpus_creation(keyphrase_dir, qfs_dir):
        logger.info('QFS corpus creation stage started')
        for split in DATASET_SPLITS:
            selected_query_distribution = list()
            selected_queries = dict()
            if split in splits_to_exclude:
                logger.info(f'Skipping {split} split')
                continue
            logger.info(f'Generating QFS corpus for {split} split')
            split_keyphrase_data = json.load(open(path.join(keyphrase_dir, split + '.query.json')))
            qfs_data = list()
            query_agnostic_data = list()
            for doc in tqdm(split_keyphrase_data):
                keyphrases = pd.DataFrame(doc['keyphrases'])
                filtered_keyphrases = keyphrases[keyphrases['score'] >= KEYPHRASE_SCORE_THRESHOLD]
                # Create summary length column
                filtered_keyphrases['summary_length'] = filtered_keyphrases['summary'].apply(len)
                sorted_keyphrases = filtered_keyphrases.sort_values(by=['score'], ascending=False)
                selected_summary_combinations = set()
                selected_queries_count = 0
                for _, keyphrase in sorted_keyphrases.iterrows():
                    summary = keyphrase['summary']
                    query = keyphrase['keyphrase']
                    if len(summary) == 0:
                        continue
                    summary_set = frozenset(summary)
                    if summary_set in selected_summary_combinations:
                        continue
                    if selected_queries_count >= MAX_QUERIES_PER_DOCUMENT:
                        break
                    selected_queries_count += 1
                    selected_queries[query] = keyphrase['score']
                    selected_summary_combinations.add(summary_set)
                    datum = {
                        'article': doc['article'],
                        'query': query,
                        'summary': summary,
                        'id': doc['id'] + '_' + str(selected_queries_count)
                    }
                    qfs_data.append(datum)
                    if selected_queries_count == 1:
                        query_agnostic_data.append(datum)
                selected_query_distribution.append(selected_queries_count)
            json.dump(qfs_data, open(path.join(qfs_dir, split + '.qfs.multiple.json'), 'w'), indent=4)
            json.dump(query_agnostic_data, open(path.join(qfs_dir, split + '.qfs.single.json'), 'w'), indent=4)
            plt.figure()
            sns.histplot(selected_query_distribution, fill=True, bins=[i for i in range(0, 4, 1)])
            plt.xlabel('Query Count')
            plt.ylabel('Documents')
            plt.title('Distribution of Queries Selected per Document')
            plt.savefig(path.join(qfs_dir, split + '.qfs.png'))
            json.dump(selected_queries, open(path.join(qfs_dir, split + '.queries.json'), 'w'), indent=4)
        logger.info('QFS corpus creation stage completed!')

    @staticmethod
    def perform_transformer_formatting(qfs_dir, summarization_dir):
        logger.info('Transformer formatting stage started')
        for split in DATASET_SPLITS:
            if split in splits_to_exclude:
                logger.info(f'Skipping {split} split')
                continue
            logger.info(f'Formatting {split} split')
            single_query_corpus = json.load(open(path.join(qfs_dir, split + '.qfs.single.json')))
            qfs_single_source = list()
            qfs_single_target = list()
            qas_single_source = list()
            qas_single_target = list()
            logger.info(f'Formatting single query corpus')
            for doc in tqdm(single_query_corpus):
                article = ' '.join(doc['article'])
                summary = ' '.join(doc['summary'])
                query = doc['query']
                qfs_single_source.append(query + ' ' + QUERY_SEPARATOR + ' ' + article)
                qas_single_source.append(article)
                qfs_single_target.append(summary)
                qas_single_target.append(summary)
            with open(path.join(summarization_dir, split + '.qfs.single.source'), 'w') as fp:
                for line in qfs_single_source:
                    fp.write(line + '\n')
            with open(path.join(summarization_dir, split + '.qfs.single.target'), 'w') as fp:
                for line in qfs_single_target:
                    fp.write(line + '\n')
            with open(path.join(summarization_dir, split + '.qas.single.source'), 'w') as fp:
                for line in qas_single_source:
                    fp.write(line + '\n')
            with open(path.join(summarization_dir, split + '.qas.single.target'), 'w') as fp:
                for line in qas_single_target:
                    fp.write(line + '\n')

            multiple_query_corpus = json.load(open(path.join(qfs_dir, split + '.qfs.multiple.json')))
            qfs_multi_source = list()
            qfs_multi_target = list()
            logger.info(f'Formatting multiple query corpus')
            for doc in tqdm(multiple_query_corpus):
                article = ' '.join(doc['article'])
                summary = ' '.join(doc['summary'])
                query = doc['query']
                qfs_multi_source.append(query + ' ' + QUERY_SEPARATOR + ' ' + article)
                qfs_multi_target.append(summary)
            with open(path.join(summarization_dir, split + '.qfs.multiple.source'), 'w') as fp:
                for line in qfs_multi_source:
                    fp.write(line + '\n')
            with open(path.join(summarization_dir, split + '.qfs.multiple.target'), 'w') as fp:
                for line in qfs_multi_target:
                    fp.write(line + '\n')
        logger.info('Transformer formatting stage completed!')

    @staticmethod
    def perform_hypothesis_tester_corpus_creation(summarization_dir, hypothesis_corpus_dir, dataset_split, summarization_corpus):
        logger.info('Hypothesis tester corpus creation stage started')
        if dataset_split == '*':
            raise Exception('Please specify a dataset split to create the tester corpus')
        random.seed(9)
        with open(path.join(summarization_dir, dataset_split + f'.{summarization_corpus}.source')) as fp:
            qas_sources = fp.read().strip().split('\n')
        with open(path.join(summarization_dir, dataset_split + f'.{summarization_corpus}.target')) as fp:
            qas_targets = fp.read().strip().split('\n')
        qas_corpus = zip(qas_sources, qas_targets)
        qas_corpus = list(qas_corpus)
        random.shuffle(qas_corpus)
        train = qas_corpus[:int(0.7 * len(qas_corpus))]
        val = qas_corpus[int(0.7 * len(qas_corpus)): int(0.85 * len(qas_corpus))]
        test = qas_corpus[int(0.85 * len(qas_corpus)):]
        with open(path.join(hypothesis_corpus_dir, f'{summarization_corpus}', 'train.source'), 'w') as source_fp, open(
                path.join(hypothesis_corpus_dir, f'{summarization_corpus}', 'train.target'), 'w') as target_fp:
            for source, target in train:
                source_fp.write(source + '\n')
                target_fp.write(target + '\n')
        with open(path.join(hypothesis_corpus_dir, f'{summarization_corpus}', 'val.source'), 'w') as source_fp, open(
                path.join(hypothesis_corpus_dir, f'{summarization_corpus}', 'val.target'), 'w') as target_fp:
            for source, target in val:
                source_fp.write(source + '\n')
                target_fp.write(target + '\n')
        with open(path.join(hypothesis_corpus_dir, f'{summarization_corpus}', 'test.source'), 'w') as source_fp, open(
                path.join(hypothesis_corpus_dir, f'{summarization_corpus}', 'test.target'), 'w') as target_fp:
            for source, target in test:
                source_fp.write(source + '\n')
                target_fp.write(target + '\n')
        logger.info('Hypothesis tester corpus creation stage completed!')

    @staticmethod
    def perform_full_corpus_creation(summarization_dir, full_corpus_dir, summarization_corpus):
        logger.info('Full corpus creation stage started')
        random.seed(9)
        with open(path.join(summarization_dir, f'train.{summarization_corpus}.source')) as fp:
            train_qas_sources = fp.read().strip().split('\n')
        with open(path.join(summarization_dir, f'train.{summarization_corpus}.target')) as fp:
            train_qas_targets = fp.read().strip().split('\n')
        train_qas_corpus = zip(train_qas_sources, train_qas_targets)
        train_qas_corpus = list(train_qas_corpus)
        random.shuffle(train_qas_corpus)
        train = train_qas_corpus[:len(train_qas_corpus) - VAL_SET_SIZE]
        val = train_qas_corpus[len(train_qas_corpus) - VAL_SET_SIZE:]
        with open(path.join(summarization_dir, f'valid.{summarization_corpus}.source')) as fp:
            val_qas_sources = fp.read().strip().split('\n')
        with open(path.join(summarization_dir, f'valid.{summarization_corpus}.target')) as fp:
            val_qas_targets = fp.read().strip().split('\n')
        val_qas_corpus = zip(val_qas_sources, val_qas_targets)
        val_qas_corpus = list(val_qas_corpus)
        test = val_qas_corpus
        logger.info(f'Writing {len(train)} train, {len(val)} val and {len(test)} test instances')
        with open(path.join(full_corpus_dir, f'{summarization_corpus}', 'train.source'), 'w') as source_fp, open(
                path.join(full_corpus_dir, f'{summarization_corpus}', 'train.target'), 'w') as target_fp:
            for source, target in train:
                source_fp.write(source + '\n')
                target_fp.write(target + '\n')
        with open(path.join(full_corpus_dir, f'{summarization_corpus}', 'val.source'), 'w') as source_fp, open(
                path.join(full_corpus_dir, f'{summarization_corpus}', 'val.target'), 'w') as target_fp:
            for source, target in val:
                source_fp.write(source + '\n')
                target_fp.write(target + '\n')
        with open(path.join(full_corpus_dir, f'{summarization_corpus}', 'test.source'), 'w') as source_fp, open(
                path.join(full_corpus_dir, f'{summarization_corpus}', 'test.target'), 'w') as target_fp:
            for source, target in test:
                source_fp.write(source + '\n')
                target_fp.write(target + '\n')
        logger.info('Full corpus creation stage completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tvt_dir", type=str, default='data/cnndm/tvt')
    parser.add_argument("-keyphrase_dir", type=str, default='data/cnndm/keyphrase')
    parser.add_argument("-qfs_dir", type=str, default='data/cnndm/qfs')
    parser.add_argument("-summarization_dir", type=str, default='data/cnndm/summarization')
    parser.add_argument("-hypothesis_corpus_dir", type=str, default='data/cnndm/hypothesis_tester')
    parser.add_argument("-full_corpus_dir", type=str, default='data/cnndm/full_corpus')
    parser.add_argument("-hypothesis_corpus", type=str, default='qas.single')
    parser.add_argument("-summarization_corpus", type=str, default='qas.single')
    parser.add_argument("-sent2vec_model", type=str, default='models/sent2vec/wiki_bigrams.bin')
    parser.add_argument("-idf_model", type=str, default='models/idf/idf_cnndm.pkl')
    parser.add_argument("-dataset_split", type=str, default='*')
    parser.add_argument('-stage', required=True, type=QfsCorpusGenerationStage, choices=list(QfsCorpusGenerationStage))
    args = parser.parse_args()

    qfs_corpus_generator = QfsCorpusGenerator()
    splits_to_exclude = []
    if args.dataset_split != '*':
        splits_to_exclude = [split for split in DATASET_SPLITS if split != args.dataset_split]

    if args.stage == QfsCorpusGenerationStage.KEYPHRASE_EXTRACTION:
        qfs_corpus_generator.perform_keyphrase_extraction(args.tvt_dir, args.keyphrase_dir, args.sent2vec_model,
                                                          args.idf_model, splits_to_exclude)
    elif args.stage == QfsCorpusGenerationStage.STATISTICS_PLOTTING:
        qfs_corpus_generator.perform_statistics_plotting(args.keyphrase_dir)
    elif args.stage == QfsCorpusGenerationStage.QFS_CORPUS_CREATION:
        qfs_corpus_generator.perform_qfs_corpus_creation(args.keyphrase_dir, args.qfs_dir)
    elif args.stage == QfsCorpusGenerationStage.TRANSFORMER_FORMATTING:
        qfs_corpus_generator.perform_transformer_formatting(args.qfs_dir, args.summarization_dir)
    elif args.stage == QfsCorpusGenerationStage.HYPOTHESIS_TESTER_CORPUS_CREATION:
        qfs_corpus_generator.perform_hypothesis_tester_corpus_creation(args.summarization_dir,
                                                                       args.hypothesis_corpus_dir, args.dataset_split,
                                                                       args.hypothesis_corpus)
    elif args.stage == QfsCorpusGenerationStage.FULL_CORPUS_CREATION:
        qfs_corpus_generator.perform_full_corpus_creation(args.summarization_dir, args.full_corpus_dir, args.summarization_corpus)
