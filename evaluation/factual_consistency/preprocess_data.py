import argparse
import logging
import json
from os import path
import os

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.DEBUG)


class FactualConsistencyPreprocessor:
    @staticmethod
    def format_for_factcc(articles_file, summaries_file, output_dir):
        logger.info(f'Formatting data for FactCC...')
        with open(articles_file) as fp:
            articles = fp.read().split('\n')
            if articles[-1] == '':
                articles = articles[:-1]
        with open(summaries_file) as fp:
            summaries = fp.read().split('\n')
            if summaries[-1] == '':
                summaries = summaries[:-1]
        assert len(articles) == len(summaries)
        factcc_data = list()
        for idx, (article, summary) in enumerate(zip(articles, summaries)):
            factcc_data.append({
                'id': idx,
                'text': article,
                'claim': summary,
                'label': 'CORRECT'
            })
        logger.info(f'Writing data to output file')
        if not path.exists(output_dir):
            os.makedirs(output_dir)
        with open(path.join(output_dir, 'data-dev.jsonl'), 'w') as fp:
            for datum in factcc_data:
                json.dump(datum, fp)
                fp.write('\n')
        logger.info(f'Formatting data for FactCC completed successfully')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-articles", type=str, required=True)
    parser.add_argument("-summaries", type=str, required=True)
    parser.add_argument("-output", type=str, required=True)
    args = parser.parse_args()
    FactualConsistencyPreprocessor.format_for_factcc(args.articles, args.summaries, args.output)
