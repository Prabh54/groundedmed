"""Answer metrics (SPEC FR-EVAL-001). A refusal (`insufficient_evidence`) counts as incorrect in
overall accuracy, is excluded from answered-only accuracy, and lowers coverage."""

ABSTAIN = "insufficient_evidence"


def _check(verdicts: list[str], gold: list[str]) -> None:
    if len(verdicts) != len(gold):
        raise ValueError("one verdict per gold label is required")
    if not verdicts:
        raise ValueError("no questions")


def accuracy(verdicts: list[str], gold: list[str]) -> float:
    _check(verdicts, gold)
    return sum(v == g for v, g in zip(verdicts, gold, strict=True)) / len(gold)


def coverage(verdicts: list[str]) -> float:
    if not verdicts:
        raise ValueError("no questions")
    return sum(v != ABSTAIN for v in verdicts) / len(verdicts)


def answered_only_accuracy(verdicts: list[str], gold: list[str]) -> float | None:
    """Accuracy over questions the system answered; None if it answered none."""
    _check(verdicts, gold)
    answered = [(v, g) for v, g in zip(verdicts, gold, strict=True) if v != ABSTAIN]
    if not answered:
        return None
    return sum(v == g for v, g in answered) / len(answered)


def majority_class_accuracy(gold: list[str], majority: str = "yes") -> float:
    """Reference: always answering "yes" (the most common PubMedQA label)."""
    return accuracy([majority] * len(gold), gold)
