import logging
import operator
import os
from typing import Any, Dict, List, Union

import numpy as np

from wolkenatlas import constants
from wolkenatlas.util import data_processing
from wolkenatlas.util import file_processing


class Embedding:
    def __init__(
            self,
            inverted_index: Dict[str, int],
            vector_space: Union[np.ndarray, Dict[str, np.ndarray]],
            **kwargs: Any
    ):
        self.inverted_index_ = inverted_index
        if isinstance(vector_space, dict):
            vector_stack = np.dstack((vector_space["input_ids"], vector_space["attention_mask"]))
            if vector_stack.shape[-1] > 2:
                vector_stack = np.dstack((vector_stack, vector_space["token_type_ids"]))
            self.vector_space_ = vector_stack

            self.fn_getitem_ = self._getitem_multi_embeddings
        else:
            self.vector_space_ = vector_space

            self.fn_getitem_ = self._getitem_single_embedding

        self.index_ = None
        self.random_state_ = np.random.RandomState(kwargs.get("random_seed", 29306))
        self.character_embeddings_ = kwargs.pop("character_embeddings", False)

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

    def index2word(self, index):
        if self.index_ is None:
            self.index_ = {idx: item for (item, idx) in self.inverted_index_.items()}
        return self.index_[index]

    def _concatenate_modalities(self, other_embedding, oov_handling="random"):
        oov_keys_in_other_emb = set(self.vocab) - set(other_embedding.vocab)
        oov_keys_in_this_emb = set(other_embedding.vocab) - set(self.vocab)
        common_keys = set(self.vocab) & set(other_embedding.vocab)

        # Build shared space
        key_idx_this = np.array([self.word2index(key) for key in common_keys])
        key_idx_other = np.array([other_embedding.word2index(key) for key in common_keys])
        space = np.hstack((self.vector_space_[key_idx_this], other_embedding.vector_space_[key_idx_other]))
        inv_idx = {key: idx for (idx, key) in enumerate(common_keys)}

        for key in oov_keys_in_other_emb:
            rnd = other_embedding.random_state_.normal(loc=0, scale=1, size=(other_embedding.dimensionality,))
            inv_idx[key] = len(inv_idx)
            vector = np.hstack((self[key], rnd))
            space = np.vstack((space, vector))

        for key in oov_keys_in_this_emb:
            rnd = self.random_state_.normal(loc=0, scale=1, size=(self.dimensionality,))
            inv_idx[key] = len(inv_idx)
            vector = np.hstack((rnd, other_embedding[key]))
            space = np.vstack((space, vector))

        return Embedding(inverted_index=inv_idx, vector_space=space)

    @staticmethod
    def _load_from_file(model_file: str, **kwargs: Any) -> "Embedding":
        emb = Embedding._empty()

        if data_processing.check_is_wolkenatlas(model_file):
            emb.inverted_index_ = file_processing.load_json(os.path.join(model_file,
                                                                         constants.INVERTED_INDEX_FILENAME))
            emb.vector_space_ = file_processing.load_vector_space(model_file)
            if emb.vector_space_.ndim > 2:
                emb.fn_getitem_ = emb._getitem_multi_embeddings
            else:
                emb.fn_getitem_ = emb._getitem_single_embedding
        else:
            file_type = kwargs.pop('file_type', None)

            if file_type is None:
                ext = os.path.splitext(model_file)[1]
                if ext not in ['.txt', '.bin', '.tgz', '.tar.gz', '.gz']:
                    raise ValueError(f'Need to specify "file_type" (e.g. "text" or "binary") if you are not '
                                     f'loading a wolkenatlas file!')
                else:
                    file_type = constants.FILE_TYPE_MAP[ext]

            loader = getattr(data_processing, f'load_{file_type}_file')
            inv_idx, vecs = loader(filename=model_file, expected_dim=kwargs.pop('expected_dim', -1),
                                   expected_vocab_size=kwargs.pop('expected_vocab_size', -1))

            emb.inverted_index_ = inv_idx
            emb.vector_space_ = vecs

        emb._finalize(**kwargs)

        return emb

    @staticmethod
    def random_model(vocab, dimensionality, random_seed=29306):
        return Embedding._create_random_model(vocab=vocab, dimensionality=dimensionality, random_seed=random_seed)

    @staticmethod
    def from_file(model_file: Union[str, List[str]], **kwargs: Any) -> "Embedding":

        if isinstance(model_file, list):
            emb = Embedding._load_from_file(model_file=model_file[0], **kwargs)
            for file in model_file[1:]:
                e = Embedding._load_from_file(model_file=file, **kwargs)

                emb = emb._concatenate_modalities(e, kwargs.get("oov_handling", "random"))
        else:
            emb = Embedding._load_from_file(model_file=model_file, **kwargs)

        return emb

    @property
    def oov(self):
        return self.oov_

    @property
    def vocab(self):
        return self.inverted_index_.keys()

    def __len__(self):
        return self.vector_space_.shape[0]

    def __contains__(self, word):
        return word in self.inverted_index_

    def __index__(self, word):
        return self.inverted_index_[word]

    def __getitem__(self, word, default=None):
        return self.fn_getitem_(word=word, default=default)

    def _getitem_multi_embeddings(self, word, default=None):
        default = default or self.oov_

        multi_key_request = isinstance(word, tuple)
        if not multi_key_request and word not in self.inverted_index_:
            data = {
                constants.INPUT_IDS_KEY: default,
                constants.ATTENTION_MASK_KEY: default
            }
            if self.vector_space_.shape[-1] == 3:
                data[constants.TOKEN_TYPE_IDS_KEY] = default
            return data

        if multi_key_request:
            word_idx = np.array(operator.itemgetter(*word)(self.inverted_index_))
        else:
            word_idx = self.inverted_index_[word]

        data = {
            constants.INPUT_IDS_KEY: self.vector_space_[word_idx, :, constants.INPUT_IDS_INDEX],
            constants.ATTENTION_MASK_KEY: self.vector_space_[word_idx, :, constants.ATTENTION_MASK_INDEX]
        }
        if self.vector_space_.shape[-1] == 3:
            data[constants.TOKEN_TYPE_IDS_KEY] = self.vector_space_[word_idx, :, constants.TOKEN_TYPE_IDS_INDEX]

        return data

    def _getitem_single_embedding(self, word, default=None):
        default = default or self.oov_

        multi_key_request = isinstance(word, tuple)

        if not multi_key_request and word not in self.inverted_index_:
            return default

        if multi_key_request:
            word_idx = np.array(operator.itemgetter(*word)(self.inverted_index_))
        else:
            word_idx = self.inverted_index_[word]

        return self.vector_space_[word_idx]
    
    @property
    def dimensionality(self):
        return self.dimensionality_

    @property
    def vocab_size(self):
        return self.vector_space_.shape[0]

    def to_file(self, filename, use_hdf=False):
        if not os.path.exists(filename):
            os.makedirs(filename)

        file_processing.save_vector_space(self.vector_space_, filename, use_hdf)
        file_processing.save_json(self.inverted_index_, os.path.join(filename, constants.INVERTED_INDEX_FILENAME))

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

    @staticmethod
    def _create_random_model(vocab, dimensionality, random_seed):
        inverted_index = {word: idx for (idx, word) in enumerate(vocab)}

        rnd = np.random.RandomState(seed=random_seed)
        vector_space = rnd.randn(len(inverted_index), dimensionality)

        return Embedding(inverted_index=inverted_index, vector_space=vector_space)

    def get_keras_tensor(self):
        pass

