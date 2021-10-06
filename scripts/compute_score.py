# adopt from https://github.com/lancopku/AMM/blob/62359a8e02fc3ae97be97e56bfb619897293c45b/model/seq.py

import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

pairs = []

with open(sys.argv[1], 'r') as f:
    target = None
    for line in f:
        if line.startswith('T-'):
            data = line.strip().split('\t')[1]
            assert target is None
            target = data
        elif line.startswith('H-'):
            data = line.strip().split('\t')[2]
            assert target is not None
            pairs.append((target.split(), data.split())) # ref, hyp
            target = None

    weights = [
        (1, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (0.33, 0.33, 0.33, 0),
        (0.25, 0.25, 0.25, 0.25)
    ]
    smoothing_fn = SmoothingFunction().method1

    scores = [np.average(
            [sentence_bleu([ref], pred, weights=w, smoothing_function=smoothing_fn) for pred, ref in pairs]) for w in weights]

    for i, score in enumerate(scores):
        print('BLEU-', i+1, score)

