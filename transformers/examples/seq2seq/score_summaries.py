import argparse
import json
import logging
import numpy as np

from utils import calculate_rouge

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.DEBUG)


def generate_scores(output_file, gold_file, save_file, metrics):
    logger.info('Starting summary score computation')
    with open(output_file) as fp:
        output_summaries = fp.read().split('\n')[:-1]
    with open(gold_file) as fp:
        gold_summaries = fp.read().split('\n')[:-1]
    scores = calculate_rouge(output_summaries, gold_summaries, newline_sep=True, rouge_keys=metrics)
    scores['output_path'] = output_file
    scores['gold_path'] = gold_file
    scores['output_summary_length'] = np.mean([len(summary.split()) for summary in output_summaries]).round(2)
    scores['gold_summary_length'] = np.mean([len(summary.split()) for summary in gold_summaries]).round(2)
    json.dump(scores, open(save_file, 'w'), indent=4)
    logger.info('Summary scores computed and saved saved successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-output", type=str, required=True)
    parser.add_argument("-gold", type=str, required=True)
    parser.add_argument("-save", type=str, required=True)
    parser.add_argument("-metrics", type=int, nargs='+', default=["rouge1", "rouge2", "rougeL"])
    args = parser.parse_args()
    generate_scores(args.output, args.gold, args.save, args.metrics)
