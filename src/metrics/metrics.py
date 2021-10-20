import editdistance


def wer(truth: str, pred: str) -> float:
    truth = truth.split()
    pred = pred.split()
    if len(truth) == 0 and len(pred) > 0:
        return 1.
    elif len(truth) == 0:
        return 0.
    distance = editdistance.distance(truth, pred)
    return distance / len(truth)


def cer(truth: str, pred: str) -> float:
    truth = list(truth)
    pred = list(pred)
    if len(truth) == 0 and len(pred) > 0:
        return 1.
    elif len(truth) == 0:
        return 0.
    distance = editdistance.distance(truth, pred)
    return distance / len(truth)
