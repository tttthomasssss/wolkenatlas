import logging
import os

import numpy as np

from wolkenatlas.util import constants
from wolkenatlas.util import data_processing
from wolkenatlas.util import file_processing


class Embedding:
    def __init__(self, model_file, init_neighbours=False, **kwargs):

        if self._check_is_wolkenatlas(model_file):
            self.inverted_index_ = file_processing.load_pickle(os.path.join(model_file,
                                                                            constants.INVERTED_INDEX_FILENAME))
            self.vector_space_ = file_processing.load_vector_space(model_file)
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

            self.inverted_index_ = inv_idx
            self.vector_space_ = emb

        self.dimensionality_ = self.vector_space_.shape[1]
        self.oov_ = kwargs.pop('oov', np.zeros((self.dimensionality_,)))

        if init_neighbours:
            pass # TODO: Load neighbours

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

    def _check_is_wolkenatlas(self, filename):
        has_vectors_file = (os.path.exists(os.path.join(filename, constants.VECTORS_FILENAME_NPY)) or
                            os.path.exists(os.path.join(filename, constants.VECTORS_FILENAME_HDF)))
        has_inv_idx_file = os.path.exists(os.path.join(filename, constants.INVERTED_INDEX_FILENAME))

        return has_vectors_file and has_inv_idx_file

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
    #emb = Embedding(model_file='/Users/thomas/research/data/glove/glove.6B.50d.txt', expected_dim=50)
    emb = Embedding(model_file='/Users/thomas/research/data/glove/glove.6B.50d.kvec')
    data = ['I like beer and pizza.', 'Pizza is actually my favourite food.', 'Pizza and pasta are my thing.']

    from wolkenatlas.preprocessing import EmbeddingsVectorizer
    vec = EmbeddingsVectorizer(embedding_model=emb)
    X = vec.transform(data)
    print(X.shape)

    #emb.to_file('/Users/thomas/research/data/glove/glove.6B.50d.kvec')

    #emb = Embedding(model_file='/Users/thomas/research/data/fastText/cc.en.300.bin', expected_dim=300, expected_vocab_size=2000000, file_type='binary')