import pytest
import VTASK
from VTASK.domains.fsm import FSMGenerator


def test_generates_valid_entry():
    gen = FSMGenerator()
    for d in range(1, 6):
        entry = gen.generate(seed=1, difficulty=d)
        assert entry.question
        assert entry.answer
        assert entry.domain == "fsm"
        assert 1 <= entry.difficulty <= 5


def test_correct_answer_scores_1():
    gen = FSMGenerator()
    for seed in range(20):
        for d in range(1, 6):
            entry = gen.generate(seed=seed, difficulty=d)
            score = gen.score_answer(entry.answer, entry)
            assert score == 1.0, f"seed={seed} d={d} answer={entry.answer!r} scored {score}"


def test_distractors_score_0():
    gen = FSMGenerator()
    for seed in range(10):
        for d in [1, 2, 4]:
            entry = gen.generate(seed=seed, difficulty=d)
            for dist in entry.distractors:
                assert gen.score_answer(dist, entry) == 0.0, \
                    f"Distractor '{dist}' incorrectly scored 1.0 at d={d} seed={seed}"


def test_garbage_scores_0():
    gen = FSMGenerator()
    entry = gen.generate(seed=42, difficulty=1)
    assert gen.score_answer("maybe", entry) == 0.0
    assert gen.score_answer("", entry) == 0.0
    assert gen.score_answer("123", entry) == 0.0


def test_determinism():
    gen = FSMGenerator()
    for d in range(1, 6):
        e1 = gen.generate(seed=77, difficulty=d)
        e2 = gen.generate(seed=77, difficulty=d)
        assert e1.question == e2.question
        assert e1.answer == e2.answer


def test_dataset_creation():
    ds = VTASK.create_dataset("fsm", size=20, seed=0)
    assert len(ds) == 20
    gen = FSMGenerator()
    for entry in ds:
        assert entry.domain == "fsm"
        assert gen.score_answer(entry.answer, entry) == 1.0
