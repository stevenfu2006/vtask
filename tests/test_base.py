import pytest
import VTASK


def test_list_domains():
    domains = VTASK.list_domains()
    assert set(domains) == {"scheduling", "dependencies", "inventory", "fsm", "spatial", "temporal"}


def test_create_dataset_invalid_domain():
    with pytest.raises(ValueError):
        VTASK.create_dataset("bogus_domain", size=5)


def test_dataset_len():
    ds = VTASK.create_dataset("fsm", size=10, seed=0)
    assert len(ds) == 10


def test_dataset_task_ids():
    ds = VTASK.create_dataset("fsm", size=5, seed=7)
    for entry in ds:
        assert entry.task_id.startswith("fsm_")


def test_to_jsonl(tmp_path):
    import json
    ds = VTASK.create_dataset("inventory", size=5, seed=1)
    path = str(tmp_path / "out.jsonl")
    ds.to_jsonl(path)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 5
    for line in lines:
        row = json.loads(line)
        assert "question" in row
        assert "answer" in row
        assert "metadata" not in row
