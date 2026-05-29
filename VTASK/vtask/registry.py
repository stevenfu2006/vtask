"""
Central registry. Maps domain name strings to generator classes.
"""
from vtask.domains.scheduling import SchedulingGenerator
from vtask.domains.dependencies import DependencyGenerator
from vtask.domains.inventory import InventoryGenerator
from vtask.domains.fsm import FSMGenerator
from vtask.domains.spatial import SpatialGenerator
from vtask.domains.temporal import TemporalGenerator

REGISTRY = {
    "scheduling": SchedulingGenerator,
    "dependencies": DependencyGenerator,
    "inventory": InventoryGenerator,
    "fsm": FSMGenerator,
    "spatial": SpatialGenerator,
    "temporal": TemporalGenerator,
}


def create_dataset(domain: str, size: int, seed: int = 42, difficulty: int | None = None):
    if domain not in REGISTRY:
        raise ValueError(f"Unknown domain '{domain}'. Available: {list(REGISTRY.keys())}")
    generator = REGISTRY[domain]()
    return generator.create_dataset(size=size, seed=seed, difficulty=difficulty)


def list_domains() -> list[str]:
    return list(REGISTRY.keys())
