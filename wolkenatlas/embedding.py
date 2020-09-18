import logging
import os

import numpy as np

from wolkenatlas import constants
from wolkenatlas.util import data_processing
from wolkenatlas.util import file_processing


class Embedding:
    def __init__(self, inverted_index, vector_space, **kwargs):
        self.inverted_index_ = inverted_index
        self.vector_space_ = vector_space

        if kwargs.pop('should_finalize', True):
            self._finalize(**kwargs)

    @staticmethod
    def _empty(): # Use at your own risk
        return Embedding(inverted_index=None, vector_space=None, should_finalize=False)

    def _finalize(self, **kwargs):
        self.dimensionality_ = self.vector_space_.shape[1]
        self.oov_ = kwargs.pop('oov', np.zeros((self.dimensionality_,)))

        if kwargs.pop('init_neighbours', False):
            pass

    @staticmethod
    def from_file(model_file, **kwargs):
        emb = Embedding._empty()

        if data_processing.check_is_wolkenatlas(model_file):
            emb.inverted_index_ = file_processing.load_pickle(os.path.join(model_file,
                                                                           constants.INVERTED_INDEX_FILENAME))
            emb.vector_space_ = file_processing.load_vector_space(model_file)
        else:
            file_type = kwargs.pop('file_type', None)

            if file_type is None:
                ext = os.path.splitext(model_file)[1]
                if ext not in ['.txt', '.bin']:
                    raise ValueError(f'Need to specify "file_type" (e.g. "text" or "binary") if you are not '
                                     f'loading a wolkenatlas file!')
                else:
                    file_type = constants.FILE_TYPE_MAP[ext]

            loader = getattr(data_processing, f'load_{file_type}_file')
            inv_idx, emb = loader(filename=model_file, expected_dim=kwargs.pop('expected_dim', -1),
                                  expected_vocab_size=kwargs.pop('expected_vocab_size', -1))

            emb.inverted_index_ = inv_idx
            emb.vector_space_ = emb

        emb._finalize(**kwargs)

        return emb

    @property
    def oov(self):
        return self.oov_

    def vocab(self):
        return self.inverted_index_.keys()

    def __len__(self):
        return self.vector_space_.shape[0]

    def __contains__(self, word):
        return word in self.inverted_index_

    def __index__(self, word):
        return self.inverted_index_[word]

    def __getitem__(self, word, default=None):
        default = default or self.oov_

        if word not in self.inverted_index_:
            return default

        return self.vector_space_[self.inverted_index_[word]]

    @property
    def dimensionality(self):
        return self.dimensionality_

    def to_file(self, filename, use_hdf=False):
        if not os.path.exists(filename):
            os.makedirs(filename)

        file_processing.save_vector_space(self.vector_space_, filename, use_hdf)
        file_processing.save_pickle(self.inverted_index_, os.path.join(filename, constants.INVERTED_INDEX_FILENAME))

    def word2index(self, word, oov=-1):
        return self.inverted_index_.get(word, oov)

    def get_pytorch_tensor(self, pad_vector=None, unk_vector=None, additional_data=None):
        return self._create_pytorch_tensor(X=self.vector_space_, pad_vector=pad_vector, unk_vector=unk_vector,
                                           additional_data=additional_data)

    def _create_pytorch_tensor(self, X, pad_vector, unk_vector, additional_data):
        try:
            import torch
        except ImportError:
            logging.error(
                'You need to have pytorch installed for using this function!\nTry running `pip install torch torchvision`!')

        if pad_vector is not None:
            X = np.vstack((X, pad_vector))
        if unk_vector is not None:
            X = np.vstack((X, unk_vector))
        if additional_data is not None:
            X = np.vstack((X, additional_data.vector_space_))

        return torch.FloatTensor(X)

    def get_keras_tensor(self):
        pass

if __name__ == '__main__':
    #emb = Embedding.from_file(model_file='/Users/thomas/research/data/glove/glove.6B.50d.txt', expected_dim=50)
    emb = Embedding.from_file(model_file='/Users/thomas/research/data/glove/glove.6B.50d.kvec')
    data = ['I like beer and pizza.', 'Pizza is actually my favourite food.', 'Pizza and pasta are my thing.']

    from wolkenatlas.preprocessing import EmbeddingsVectorizer
    vec = EmbeddingsVectorizer(embedding_model=emb)
    X = vec.transform(data)
    print(X.shape)

    #emb.to_file('/Users/thomas/research/data/glove/glove.6B.50d.kvec')

    #emb = Embedding(model_file='/Users/thomas/research/data/fastText/cc.en.300.bin', expected_dim=300, expected_vocab_size=2000000, file_type='binary')