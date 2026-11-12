"""Analytical checks independent of the review corpus."""
import math
import unittest
from naive_bayes import train_nb, score_doc_label, classify_nb, accuracy


class NaiveBayesChecks(unittest.TestCase):
    def test_hand_calculated_probabilities(self):
        model = train_nb([['good', 'good', 'film'], ['bad', 'film']], ['pos', 'neg'])
        # V=3; positive denominator=3+3, negative denominator=2+3.
        self.assertAlmostEqual(math.exp(score_doc_label(['good'], 'pos', model)), .5*3/6)
        self.assertAlmostEqual(math.exp(score_doc_label(['good'], 'neg', model)), .5*1/5)
        self.assertAlmostEqual(score_doc_label(['good', 'good'], 'pos', model),
                               math.log(.5)+2*math.log(3/6))
        for c in model.classes:
            self.assertAlmostEqual(sum(math.exp(p) for p in model.log_likelihood[c].values()), 1.)

    def test_unknown_words_and_prior(self):
        model = train_nb([['x'], ['x'], ['y']], ['pos', 'pos', 'neg'])
        self.assertNotIn('unseen', model.vocabulary)
        self.assertEqual(classify_nb(['unseen'], model), 'pos')
        self.assertEqual(score_doc_label(['unseen'], 'pos', model), model.log_prior['pos'])

    def test_generic_classes_and_bad_inputs(self):
        model = train_nb([['a'], ['b'], ['c']], ['one', 'two', 'three'])
        self.assertEqual(classify_nb(['b'], model), 'two')
        self.assertEqual(accuracy(['x', 'y'], ['x', 'x']), .5)
        with self.assertRaises(ValueError):
            train_nb([['a']], [])
        with self.assertRaises(ValueError):
            train_nb([['a']], ['pos'], alpha=0)
        with self.assertRaises(ValueError):
            accuracy([], [])


if __name__ == '__main__':
    unittest.main()
