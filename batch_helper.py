import math

def batches(batch_size, feature, labels):
    """
    create batches of features and labels
    :param batch_size: The batch size
    :param feature: List of features
    :param labels: List of labels
    :return: Batches of(Features, Labels)
    """
    assert len(feature) == len(labels)
    output_batches = []

    sample_size = len(feature)

    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [feature[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
    return output_batches