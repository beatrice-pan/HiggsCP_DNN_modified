import numpy as np


def calculate_deltas_unsigned(expected, actual, num_classes):
    deltas = np.min(
        np.stack(
            [np.abs(expected - actual), (num_classes - np.abs(expected - actual))]
        ), axis=0)

    return deltas

# corrected, so the first and last class are the same class
# OK for using in range (0, 2 pi) and trygonometric function
# not OK otherwise!
def calculate_deltas_signed(expected, actual, num_classes):
    deltas = actual - expected
    deltas -= (num_classes-1) * (deltas > (num_classes // 2))
    deltas += (num_classes-1) * (deltas <= (-num_classes // 2))

    return deltas

# by ERW
def calculate_deltas_signed_pi(expected, actual):
    deltas = actual - expected
    deltas -= np.pi * (deltas > np.pi)
    deltas += np.pi * (deltas <= -np.pi)

    return deltas

# by J.Kurek
def calculate_deltas_signed_pi_topo(expected, actual):
    # Unsigned
    deltas = np.minimum(np.abs(actual - expected), 2 * np.pi - np.abs(actual - expected))
    deltas *= np.sign(np.sin(expected - actual))
    return deltas
