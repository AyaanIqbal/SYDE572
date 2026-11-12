"""Run from any directory; outputs are saved beside this script."""
from pathlib import Path
from collections import Counter
import csv
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from naive_bayes import (read_documents, train_nb, score_doc_label,
                         classify_nb, classify_documents, accuracy)

ROOT = Path(__file__).resolve().parent


def main():
    for folder in ('results', 'plots'):
        (ROOT / folder).mkdir(exist_ok=True)
    source = ROOT / 'data/all_sentiment_shuffled.txt'
    docs, labels = read_documents(source)
    metadata = [line.split()[:3] for line in source.read_text().splitlines()]
    split = int(.8 * len(docs))
    train_docs, test_docs = docs[:split], docs[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    model = train_nb(train_docs, train_labels)
    predictions = classify_documents(test_docs, model)
    result = accuracy(test_labels, predictions)
    correct = sum(a == b for a, b in zip(test_labels, predictions))
    n = len(test_docs)
    z = 1.959963984540054
    center = (result + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(result*(1-result)/n + z*z/(4*n*n))/(1+z*z/n)
    confusion = {a: {b: sum(t == a and p == b for t, p in zip(test_labels, predictions))
                     for b in model.classes} for a in model.classes}
    sanity = {word: {c: math.exp(score_doc_label([word], c, model))
                     for c in model.classes} for word in ('great', 'bad')}
    assert sanity['great']['pos'] > sanity['great']['neg']
    assert sanity['bad']['neg'] > sanity['bad']['pos']
    unknown_prediction = classify_nb(['a', 'top-quality', 'performance'], model)
    metrics = dict(total=len(docs), training=split, evaluation=n,
                   training_labels=dict(Counter(train_labels)),
                   evaluation_labels=dict(Counter(test_labels)),
                   vocabulary=len(model.vocabulary), alpha=model.alpha,
                   correct=correct, accuracy=result,
                   majority_baseline=max(Counter(test_labels).values())/n,
                   wilson_95=[center-half, center+half],
                   confusion=confusion, sanity_checks=sanity,
                   unknown_token_check=unknown_prediction)
    (ROOT/'results/metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    errors = []
    with (ROOT/'results/predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['line', 'topic', 'document_id', 'true', 'predicted', 'pos_minus_neg'])
        for i, (doc, true, pred) in enumerate(zip(test_docs, test_labels, predictions)):
            delta = score_doc_label(doc, 'pos', model)-score_doc_label(doc, 'neg', model)
            topic, _, identifier = metadata[split+i]
            writer.writerow([split+i+1, topic, identifier, true, pred, delta])
            if true != pred:
                contributions = {w: count*(model.log_likelihood[pred][w]-model.log_likelihood[true][w])
                                 for w, count in Counter(doc).items() if w in model.vocabulary}
                errors.append(dict(line=split+i+1, topic=topic, document_id=identifier,
                                   true=true, predicted=pred, tokens=len(doc), margin=abs(delta),
                                   text=' '.join(doc),
                                   strongest_wrong_evidence=sorted(contributions.items(), key=lambda x:-x[1])[:8]))
    short_errors = sorted((e for e in errors if e['tokens'] <= 80), key=lambda e:-e['margin'])[:6]
    (ROOT/'results/error_examples.json').write_text(json.dumps(short_errors, indent=2)+'\n')
    curve = []
    for tenth in range(1, 11):
        size = int(split*tenth/10)
        current = model if tenth == 10 else train_nb(train_docs[:size], train_labels[:size])
        curve.append((size, accuracy(test_labels, classify_documents(test_docs, current))))
    with (ROOT/'results/learning_curve.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['training_reviews', 'accuracy'])
        writer.writerows(curve)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(*zip(*curve), marker='o')
    axes[0].axhline(metrics['majority_baseline'], color='gray', linestyle='--', label='Test majority baseline')
    axes[0].set(xlabel='Training reviews', ylabel='Test accuracy', title='Learning curve', ylim=(.45, 1))
    axes[0].legend()
    matrix = [[confusion[a][b] for b in model.classes] for a in model.classes]
    axes[1].imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, matrix[i][j], ha='center', va='center',
                         color='white' if i == j else 'black')
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=model.classes,
                yticklabels=model.classes, xlabel='Predicted', ylabel='True', title='Test confusion matrix')
    fig.savefig(ROOT/'plots/evaluation.png', dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print('\nError examples:', json.dumps(short_errors, indent=2))


if __name__ == '__main__':
    main()
