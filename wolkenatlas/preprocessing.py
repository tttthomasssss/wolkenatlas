from collections import Counter
import logging

import numpy as np

from wolkenatlas import encoder


class EmbeddingsVectorizer():
    def __init__(self, embedding_model, encoder_model='average', tokenizer=lambda x: x.split(), vocab=None, min_df=1,
                 lowercase=True, transform_to_tensor_type='numpy'):
        self.embedding_model_ = embedding_model
        self.encoder_model_ = getattr(encoder, f'{encoder_model}_encoder')
        self.tokenizer_ = tokenizer
        self.vocab_ = vocab
        self.min_df_ = min_df
        self.lowercase_ = lambda x: x.lower() if lowercase else lambda x: x
        self.transform_to_tensor_type_ = getattr(self, f'_to_{transform_to_tensor_type}')

    def fit(self, documents):
        freq_table = self._fit_freq_table(documents=documents)

        if self.vocab_ is None:
            self.vocab_ = set()
            self._filter_extremes(freq_table=freq_table)
        else:
            self._filter_extremes_with_vocab(freq_table=freq_table)

    def transform(self, documents):
        data = []

        for doc in documents:
            x_doc = []
            for token in self.tokenizer_(doc):
                if self.lowercase_(token) in self.vocab_:
                    x_doc.append(self.embedding_model_[self.lowercase_(token)])
            transformed_doc = self.encoder_model_(x_doc)

            # TODO: Need better handling when a document can't be transformed at all (i.e. all items are not within the specified vocabulary)
            if len(transformed_doc) <= 0:
                transformed_doc = self.embedding_model_.oov

            data.append(transformed_doc)

        return self.transform_to_tensor_type_(data)

    def _to_numpy(self, data):
        return np.array(data)

    def _to_tf(self, data):
        try:
            import tensorflow as tf
        except ImportError as ex:
            logging.error(f'You need to install tensorflow to use this function!')
            raise ex

        return tf.convert_to_tensor(self._to_numpy(data), dtype=np.float32)

    def _to_pytorch(self, data):
        try:
            import torch
        except ImportError as ex:
            logging.error('You need to install pytorch to use this function!')
            raise ex

        return torch.FloatTensor(self._to_numpy(data))

    def _fit_freq_table(self, documents):
        freqs = Counter()
        for doc in documents:
            for token in self.tokenizer_(doc):
                freqs[self.lowercase_(token)] += 1

        return freqs

    def _filter_extremes(self, freq_table):
        for token in freq_table.elements():
            if freq_table[token] >= self.min_df_:
                self.vocab_.add(token)

    def _filter_extremes_with_vocab(self, freq_table):
        for token in freq_table.elements():
            if token in self.vocab_ and freq_table[token] < self.min_df_:
                self.vocab_.remove(token)