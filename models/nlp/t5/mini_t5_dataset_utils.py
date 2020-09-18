import random
import numpy as np


def random_spans_noise_mask(length, noise_density, mean_noise_span_length=3.0):
    """Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(
             num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean array with shape [length]
    """
    orig_length = length
    # increase length to avoid degeneracy
    length = max(length, 2)
    num_noise_tokens = int(round(float(length) * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(round(float(num_noise_tokens) / mean_noise_span_length))
    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of tokens randomly into non-empty spans.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a array with shape [num_segments] containing positive integers that add
            up to num_items
        """
        x = [1] * (num_segments - 1) + [0] * (num_items - num_segments)
        x = np.array(x)  # convert
        np.random.shuffle(x)
        first_in_segment = np.pad(x, (1, 0))  # padding a zero to the start
        segment_id = np.cumsum(first_in_segment)

        segment_length = []
        l = r = 0
        while r < len(segment_id):
            if segment_id[r] != segment_id[l]:
                segment_length.append(r - l)
                l = r
            r += 1
        segment_length.append(r - l)  # calculate the last sequence
        return np.array(segment_length)

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    def _interleave(lengths_a, lengths_b):
        interleaved_lengths = np.empty((lengths_a.size + lengths_b.size), dtype=lengths_a.dtype)
        interleaved_lengths[0::2] = lengths_a
        interleaved_lengths[1::2] = lengths_b
        return interleaved_lengths

    interleaved_span_lengths = _interleave(nonnoise_span_lengths, noise_span_lengths)
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.array([0 if i not in span_starts else 1 for i in range(length)])
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)
    return is_noise[:orig_length]

def noise_span_to_unique_sentinel(tokens, noise_mask, vocabulary=None):
    """Replace each run of consecutive noise tokens with a different sentinel.
    The idea here is to be able to align the dropped spans in the inputs
    with the markers in the targets.
    We want to generate training examples like
    "We hold X to be Y that" -> "X these truths Y self evident Z"
    Sentinels assigned in decreasing order within the sequence starting at
    vocabulary.size - 1.  That is, we appropriate the last tokens in the
    vocabulary for additional use as sentinels.
    TODO(noam): we may want to try enlarging the vocabulary and leaving room
    for the sentinels instead.  However, this requires enlarging the embedding
    tables in the model, so that is a bigger change.
    Args:
        tokens: a 1d integer numpy array
        noise_mask: a boolean numpy array with the same shape as tokens
        vocabulary: a vocabulary.Vocabulary
    Returns:
        a numpy array with the same shape and dtype as tokens
    """
    # TODO: figure out proper vocab size
    # vocab_size = vocabulary.vocab_size
    vocab_size = 300
    prev_token_is_noise = np.pad(noise_mask[:-1], [[1, 0]])
    # first token in noise spans
    first_noise_tokens = np.logical_and(noise_mask, np.logical_not(prev_token_is_noise))
    # subsequent tokens in noise spans
    subsequent_noise_tokens = np.logical_and(noise_mask, prev_token_is_noise)
    # TODO: replace sentinel with proper id
    # using sentinel from the last of vocabulary
    sentinel = vocab_size - np.cumsum(first_noise_tokens.astype(int))
    tokens = np.where(first_noise_tokens, sentinel, tokens)
    # boolean mask
    sentineled_tokens = tokens[np.logical_not(subsequent_noise_tokens)]
    return sentineled_tokens

def nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocabulary=None):
    return noise_span_to_unique_sentinel(tokens, np.logical_not(noise_mask), vocabulary)

def build_sample(tokens, vocabulary):
    """
    Args:
        tokens: a 1d integer numpy array
        noise_mask: a boolean numpy array with the same shape as tokens
        vocabulary: a vocabulary.Vocabulary
    Returns:
        a numpy array with the same shape and dtype as tokens
    """
    noise_mask = random_spans_noise_mask(len(tokens), noise_density)
    inputs = noise_span_to_unique_sentinel(tokens, noise_mask, vocabulary)
    targets = nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocabulary)
    return {'inputs': inputs, 'targets': targets}

if __name__ == "__main__":
    np.random.seed(123)
    spans_mask = random_spans_noise_mask(50, 0.15)
    noise_span_to_unique_sentinel(np.array([1] * 50), spans_mask)
    nonnoise_span_to_unique_sentinel(np.array([1] * 50), spans_mask)
    print(build_sample(np.array([1] * 50), 0.15))