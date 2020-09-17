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


if __name__ == "__main__":
    np.random.seed(123)
    spans_mask = random_spans_noise_mask(100, 0.15)