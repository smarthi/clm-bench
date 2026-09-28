import numpy as np

from clm_bench import metrics as M


def test_softmax_and_nll_match_hand_computation():
    z = [np.array([2.0, 0.0, 0.0])]
    p = M.softmax(z[0])
    assert np.isclose(p.sum(), 1.0)
    assert np.isclose(M.nll(z, np.array([0])), -np.log(p[0]))


def test_brier_perfect_and_worst():
    assert M.brier([np.array([1.0, 0.0])], np.array([0])) == 0.0
    assert M.brier([np.array([0.0, 1.0])], np.array([0])) == 2.0


def test_ece_zero_when_confidence_equals_accuracy():
    conf = np.array([0.8] * 10)
    correct = np.array([1] * 8 + [0] * 2)
    e, mce = M.ece(conf, correct)
    assert np.isclose(e, 0.0) and np.isclose(mce, 0.0)


def test_ece_overconfident():
    conf = np.array([0.95] * 10)
    correct = np.array([1] * 5 + [0] * 5)
    e, _ = M.ece(conf, correct)
    assert np.isclose(e, 0.45)


def test_temperature_recovers_known_value():
    # Draw labels from softmax(z / 3); logits z are then overconfident by T=3.
    rng = np.random.default_rng(0)
    logits, gold = [], []
    for _ in range(4000):
        z = rng.normal(0, 4, size=4)
        logits.append(z)
        gold.append(rng.choice(4, p=M.softmax(z, 3.0)))
    t = M.fit_temperature(logits, np.array(gold))
    assert 2.6 < t < 3.4, t


def test_variable_k():
    logits = [np.array([1.0, 0.0, -1.0]), np.array([0.0, 2.0, 0.0, 0.0, 0.0])]
    s = M.summarize(logits, np.array([0, 1]))
    assert s["accuracy"] == 1.0
    assert np.isclose(s["chance_accuracy"], (1 / 3 + 1 / 5) / 2)
