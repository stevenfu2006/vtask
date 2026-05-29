import pytest
import VTASK
from VTASK.domains.dependencies import DependencyGenerator


def test_generates_valid_entry():
    gen = DependencyGenerator()
    for d in range(1, 6):
        entry = gen.generate(seed=1, difficulty=d)
        assert entry.question
        assert entry.answer
        assert entry.domain == "dependencies"


def test_correct_answer_scores_1():
    gen = DependencyGenerator()
    for seed in range(20):
        for d in range(1, 6):
            entry = gen.generate(seed=seed, difficulty=d)
            score = gen.score_answer(entry.answer, entry)
            assert score == 1.0, f"seed={seed} d={d} answer={entry.answer!r} scored {score}"


def test_distractors_score_0():
    gen = DependencyGenerator()
    for seed in range(10):
        for d in [1, 2, 4]:
            entry = gen.generate(seed=seed, difficulty=d)
            for dist in entry.distractors:
                assert gen.score_answer(dist, entry) == 0.0, \
                    f"Distractor '{dist}' scored 1.0 at d={d} seed={seed}"


def test_garbage_scores_0():
    gen = DependencyGenerator()
    entry = gen.generate(seed=42, difficulty=2)
    assert gen.score_answer("not a number", entry) == 0.0
    assert gen.score_answer("", entry) == 0.0
    assert gen.score_answer("-999", entry) == 0.0


def test_determinism():
    gen = DependencyGenerator()
    for d in range(1, 6):
        e1 = gen.generate(seed=77, difficulty=d)
        e2 = gen.generate(seed=77, difficulty=d)
        assert e1.question == e2.question
        assert e1.answer == e2.answer


def test_dataset_creation():
    ds = VTASK.create_dataset("dependencies", size=20, seed=0)
    assert len(ds) == 20
    gen = DependencyGenerator()
    for entry in ds:
        assert entry.domain == "dependencies"
        assert gen.score_answer(entry.answer, entry) == 1.0
