import logging

import numpy as np


def _check_has_header(line, encoding='utf-8', sep=' '):
    try:
        line = line.decode(encoding).strip() if isinstance(line, bytes) else line.strip()
    except UnicodeDecodeError:
        return False

    return True if len(line.split(sep)) == 2 else False


def process_header(header, encoding='utf-8', sep=' '):
    header = header.decode(encoding).strip() if isinstance(header, bytes) else header.strip()
    vocab_size, vector_dim = header.split(sep)

    return int(vocab_size), int(vector_dim)


def load_binary_file(filename, encoding='utf-8', expected_dim=-1, expected_vocab_size=-1,
                     expected_dtype=np.float32, header_sep=' ', output_dtype=np.float32):
    with open(filename, 'rb') as in_file:
        line = next(in_file)
        has_header = _check_has_header(line, encoding=encoding, sep=header_sep)
        if has_header:
            expected_vocab_size, expected_dim = process_header(line, encoding=encoding, sep=header_sep)
        else:
            in_file.seek(0)

        if not has_header and expected_dim < 0 and expected_vocab_size < 0:
            raise ValueError(f'If the file does not contain a header (has_header={has_header}), then expected_dim '
                             f'and expected_vocab_size must be passed manually (expected_dim={expected_dim}, '
                             f'expected_vocab_size={expected_vocab_size})!')

        inv_idx = {}
        data = []
        binary_len = np.dtype(expected_dtype).itemsize * expected_dim

        # thank you gensim!
        for idx in range(expected_vocab_size):
            if idx % 10000 == 0: logging.debug(f'{idx} / {expected_vocab_size} items processed!')
            word = []
            while True:
                ch = in_file.read(1)
                if ch == b' ':
                    break
                if ch != b'\n':
                    word.append(ch)
            word = b''.join(word).decode('utf-8')
            weights = np.fromstring(in_file.read(binary_len), dtype=np.float32)

            if weights.shape[0] != expected_dim:
                logging.warning(f'Dimension of extracted weight vector (len(weights)={weights.shape[0]}) does not '
                                f'match the expected dimensionality (expected_dim={expected_dim}), skipping weights '
                                f'for vocabulary item="{word}"!')
            else:
                if word in inv_idx:
                    logging.warning(f'"{word}" found in vocabulary twice, keeping only first occurrence!')
                else:
                    inv_idx[word] = len(inv_idx)
                    data.append(weights)

        return inv_idx, np.array(data).astype(output_dtype)


def load_text_file(filename, encoding='utf-8', expected_dim=-1,  header_sep=' ', pos_separator='__',
                   output_dtype=np.float32, data_sep=' ', strip_pos_tags=False, **_):
    with open(filename, encoding=encoding) as in_file:
        line = next(in_file)
        has_header = _check_has_header(line, encoding=encoding, sep=header_sep)
        if has_header:
            expected_vocab_size, expected_dim = process_header(line, encoding=encoding, sep=header_sep)
        else:
            in_file.seek(0)

        if not has_header and expected_dim < 0:
            raise ValueError(f'If the file does not contain a header (has_header={has_header}), then expected_dim '
                             f'must be passed manually (expected_dim={expected_dim})!')

        inv_idx = {}
        data = []
        for idx, line in enumerate(in_file, 1):
            if idx % 10000 == 0: logging.debug(f'{idx} lines processed!')

            parts = line.rstrip().split(data_sep)
            word = parts[0]
            if (strip_pos_tags):
                word = word.split(pos_separator)[0]

            try:
                weights = list(map(lambda x: float(x), parts[1:]))
            except ValueError as ex:
                logging.warning(f'Failed to parse line {idx} (word={word}): {ex}')

            if len(weights) != expected_dim:
                logging.warning(f'Dimension of extracted weight vector (len(weights)={len(weights)} does not '
                                f'match the expected dimensionality (expected_dim={expected_dim}), skipping weights '
                                f'for vocabulary item="{word}"!')
            else:
                if word in inv_idx:
                    logging.warning(f'"{word}" found in vocabulary twice, keeping only first occurrence!')
                else:
                    inv_idx[word] = len(inv_idx)
                    data.append(np.array(weights))

    return inv_idx, np.array(data).astype(output_dtype)