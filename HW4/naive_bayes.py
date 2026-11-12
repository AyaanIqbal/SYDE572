"""Multinomial Naive Bayes implemented with the Python standard library."""
from collections import Counter
from dataclasses import dataclass
import math


@dataclass
class NBModel:
    classes: tuple
    vocabulary: set
    log_prior: dict
    log_likelihood: dict
    counts: dict
    alpha: float


def read_documents(doc_file):
    documents, labels = [], []
    with open(doc_file, encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f'Missing metadata on line {number}')
            documents.append(fields[3:])
            labels.append(fields[1])
    return documents, labels


def train_nb(documents, labels, alpha=1.0):
    if not documents or len(documents) != len(labels):
        raise ValueError('Require nonempty, equally sized documents and labels')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    classes = tuple(sorted(set(labels)))
    counts = {label: Counter() for label in classes}
    frequencies = Counter(labels)
    for document, label in zip(documents, labels):
        counts[label].update(document)
    vocabulary = set().union(*(set(c) for c in counts.values()))
    if not vocabulary:
        raise ValueError('Training vocabulary must be nonempty')
    log_prior = {c: math.log(frequencies[c] / len(labels)) for c in classes}
    log_likelihood = {}
    for c in classes:
        denominator = sum(counts[c].values()) + alpha * len(vocabulary)
        log_likelihood[c] = {w: math.log((counts[c][w] + alpha) / denominator)
                             for w in vocabulary}
    return NBModel(classes, vocabulary, log_prior, log_likelihood, counts, alpha)


def score_doc_label(document, label, model):
    """Log score log P(c) + sum_w count(w,d) log P(w|c).

    Ignore tokens outside the training vocabulary. The multinomial coefficient,
    constant across classes, is omitted. This is an unnormalized class score,
    not a posterior probability.
    """
    return model.log_prior[label] + sum(
        count * model.log_likelihood[label][word]
        for word, count in Counter(document).items() if word in model.vocabulary)


def classify_nb(document, model):
    # Sorted class order resolves exact ties deterministically.
    return max(model.classes, key=lambda c: score_doc_label(document, c, model))


def classify_documents(docs, model):
    return [classify_nb(doc, model) for doc in docs]


def accuracy(true_labels, guessed_labels):
    if not true_labels or len(true_labels) != len(guessed_labels):
        raise ValueError('Require nonempty, equally sized label lists')
    return sum(a == b for a, b in zip(true_labels, guessed_labels)) / len(true_labels)
