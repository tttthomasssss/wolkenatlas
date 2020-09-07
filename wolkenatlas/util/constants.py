VECTORS_FILENAME_HDF = 'X.hdf'
VECTORS_FILENAME_NPY = 'X.npy'
INVERTED_INDEX_FILENAME = 'inverted_index.pkl'

FILE_TYPE_MAP = {
    '.txt': 'text',
    '.bin': 'binary'
}

COMPOSITION_FUNCTION_DIM_MULTIPLIER = {
    'sum': 1,
    'average': 1,
    'max': 1,
    'concatenate_average_max': 2,
    'concatenate_sum_max': 2,
    'concatenate_average_min': 2,
    'concatenate_sum_min': 2,
    'concatenate_min_max': 2
}

FASTTEXT_CC_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/'
FASTTEXT_WIKI_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/'
FASTTEXT_ALIGNED_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/'

FASTTEXT_CC_VECTOR_NAME_TEMPLATE = 'cc.{}.300.vec.gz'
FASTTEXT_WIKI_VECTOR_NAME_TEMPLATE = 'wiki.{}.vec'
FASTTEXT_ALIGNED_VECTOR_NAME_TEMPLATE = 'wiki.{}.align.vec'

FASTTEXT_TRAINING_CORPUS_URL_MAP = {
    'cc': FASTTEXT_CC_URL,
    'wiki': FASTTEXT_WIKI_URL,
    'aligned': FASTTEXT_ALIGNED_URL
}

FASTTEXT_TRAINING_CORPUS_VECTOR_NAME_MAP = {
    'cc': FASTTEXT_CC_VECTOR_NAME_TEMPLATE,
    'wiki': FASTTEXT_WIKI_VECTOR_NAME_TEMPLATE,
    'aligned': FASTTEXT_ALIGNED_VECTOR_NAME_TEMPLATE
}