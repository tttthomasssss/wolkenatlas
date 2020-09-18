from argparse import ArgumentParser
import csv
import logging
import os
import urllib

from wolkenatlas import constants

parser = ArgumentParser()
parser.add_argument('-m', '--vector-model', type=str, help='vector model to use.', required=True, choices=['fasttext',
                                                                                                          'glove',
                                                                                                          'word2vec'])
parser.add_argument('-l', '--language', type=str, help='2-letter ISO 639-1 Code for the given language (e.g. "de" for '
                                                       'German or "es" for Spanish).', required=True)
parser.add_argument('-tc', '--training-corpus', type=str, help='Training corpus used for vectors.')

def validate_args(args):
    return True


def download_fasttext_model(language, training_corpus):
    vector_file = constants.FASTTEXT_TRAINING_CORPUS_VECTOR_NAME_MAP[training_corpus].format(language)
    url = constants.FASTTEXT_TRAINING_CORPUS_URL_MAP[training_corpus]

    with urllib.requst.urlopen(url) as ft:
        meta = ft.info()
        logging.info(f'Downloading data from {url} ({round(int(meta["Content-Length"])/1000)} kb)')

        xxx = ft.read()


if __name__ == '__main__':
    args = parser.parse_args()

    if validate_args(args):

        download_model()

    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Load experiment id file
    if args.experiment_file is not None:
        with open(os.path.join(PACKAGE_PATH, 'resources', 'analysis', args.experiment_file), 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            experiments = []

            for line in csv_reader:
                experiments.append(line)

    if args.action == 'calculate_total_lexical_overlap':
        calculate_total_lexical_overlap(training_data_file=os.path.join(args.input_path, args.input_file),
                                        evaluation_data_file=os.path.join(args.input_path_2, args.input_file_2),
                                        output_file=os.path.join(args.output_path, args.output_file))
    elif args.action == 'calculate_average_per_item_overlap':
        calculate_average_per_item_overlap(training_data_file=os.path.join(args.input_path, args.input_file),
                                           evaluation_data_file=os.path.join(args.input_path_2, args.input_file_2),
                                           output_file=os.path.join(args.output_path, args.output_file))
    elif args.action == 'estimate_keyword_coverage':
        estimate_keyword_coverage(training_data_file=os.path.join(args.input_path, args.input_file),
                                  evaluation_data_file=os.path.join(args.input_path_2, args.input_file_2),
                                  output_file=os.path.join(args.output_path, args.output_file))
    elif args.action == 'extract_keywords_by_mutual_information':
        extract_keywords_by_mutual_information(training_data_file=os.path.join(args.input_path, args.input_file),
                                               evaluation_data_file=os.path.join(args.input_path_2, args.input_file_2))
    elif args.action == 'calculate_lexical_diversity':
        calculate_lexical_diversity(training_data_file=os.path.join(args.input_path, args.input_file),
                                    evaluation_data_file=os.path.join(args.input_path_2, args.input_file_2))
    elif args.action == 'calculate_token_entropy':
        calculate_token_entropy(training_data_file=os.path.join(args.input_path, args.input_file),
                                evaluation_data_file=os.path.join(args.input_path_2, args.input_file_2),
                                output_file=os.path.join(args.output_path, args.output_file),
                                min_freq=args.min_token_frequency)
    elif args.action == 'calculate_token_ngram_entropy':
        for exp_id, (training_file, output_sub_path, min_frequency, ngram_span) in enumerate(experiments, 1):
            logging.info(f'Running Experiment with ID={exp_id}...')
            if not os.path.exists(os.path.join(args.output_path, output_sub_path)):
                os.makedirs(os.path.join(args.output_path, output_sub_path))
            calculate_token_ngram_entropy(training_data_file=os.path.join(args.input_path, training_file),
                                          output_path=os.path.join(args.output_path, output_sub_path),
                                          ngram_spans=list(map(lambda x: int(x), ngram_span.split('-'))),
                                          min_freq=int(min_frequency))
    elif args.action == 'calculate_information_gain':
        for exp_id, (training_file, output_sub_path, min_frequency, ngram_span) in enumerate(experiments, 1):
            logging.info(f'Running Experiment with ID={exp_id}...')
            if not os.path.exists(os.path.join(args.output_path, output_sub_path)):
                os.makedirs(os.path.join(args.output_path, output_sub_path))
            calculate_information_gain(training_data_file=os.path.join(args.input_path, training_file),
                                       output_path=os.path.join(args.output_path, output_sub_path),
                                       ngram_spans=list(map(lambda x: int(x), ngram_span.split('-'))),
                                       min_freq=int(min_frequency))
        # TODO: for a trained model, see if the right intents are predicted for just the most salient keywords --> opposite of avoiding lexical overlap
        # TODO: train a model with a single intent, consisting of the vocab of the most salient keywords only and run on test set.
        # TODO: average similarity per class, BoW and NN encoder
        # TODO: Mutual INformation / Chi-Squre between a word (vector) and its class label
        # TODO: Token Entropy: collect all classes for a token and calculate its
        # TODO: For the "few tokens per intent task" use all extracted, only nouns/verbs
        # TODO: Train only on intent tokens
        # TODO: BrownClustering to get sentence embeddings for intent utterances
    else:
        raise ValueError(f'Action "{args.action}" not supported!')