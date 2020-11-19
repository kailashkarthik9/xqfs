import argparse
import logging
import pickle
from enum import Enum

from query_generation.keyphrase_extraction.launch import load_local_embedding_distributor, extract_keyphrases
from query_generation.keyphrase_extraction.postagging import PosTaggingSpacy
import json
from os import path
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.DEBUG)

DATASET_SPLITS = [
    'test',
    'valid',
    'train',
]


class QfsCorpusGenerationStage(Enum):
    KEYPHRASE_EXTRACTION = 'keyphrase_extraction'


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
                keyphrases, scores, synsets = extract_keyphrases(self.embedding_distributor, self.pos_tagger, self.idf_model,
                                                        summary_text, 10, 'en')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tvt_dir", type=str, default='data/cnndm/tvt')
    parser.add_argument("-keyphrase_dir", type=str, default='data/cnndm/keyphrase')
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
        qfs_corpus_generator.perform_keyphrase_extraction(args.tvt_dir, args.keyphrase_dir, args.sent2vec_model, args.idf_model, splits_to_exclude)
