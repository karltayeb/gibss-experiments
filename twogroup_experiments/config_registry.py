from __future__ import annotations

from typing import Iterable

from core import HASH_KEY, MethodSpec, SimulationSpec, dehydrate_hashed
from utils import BatchSpec


class ConfigRegistry:
    def __init__(self) -> None:
        self._simulations_by_name: dict[str, SimulationSpec] = {}
        self._simulation_hashes: dict[str, SimulationSpec] = {}
        self._methods_by_hash: dict[str, MethodSpec] = {}
        self._batches_by_name: dict[str, BatchSpec] = {}
        self._batch_hashes: dict[str, BatchSpec] = {}

    @property
    def simulations(self) -> tuple[SimulationSpec, ...]:
        return tuple(self._simulations_by_name.values())

    @property
    def methods(self) -> tuple[MethodSpec, ...]:
        return tuple(self._methods_by_hash.values())

    @property
    def batches(self) -> tuple[BatchSpec, ...]:
        return tuple(self._batches_by_name.values())

    def _item_hash(self, item) -> str:
        return dehydrate_hashed(item)[HASH_KEY]

    def _register_named(self, *, kind: str, item, by_name: dict, by_hash: dict):
        item_hash = self._item_hash(item)
        existing = by_name.get(item.name)
        if existing is not None:
            if self._item_hash(existing) != item_hash:
                raise ValueError(
                    f"Conflicting {kind} registration for name {item.name!r}."
                )
            return existing
        hashed_existing = by_hash.get(item_hash)
        if hashed_existing is not None and hashed_existing.name != item.name:
            raise ValueError(
                f"Conflicting {kind} registrations share spec hash {item_hash}: "
                f"{hashed_existing.name!r} vs {item.name!r}."
            )
        by_name[item.name] = item
        by_hash[item_hash] = item
        return item

    def register_simulations(
        self, simulations: Iterable[SimulationSpec]
    ) -> tuple[SimulationSpec, ...]:
        return tuple(
            self._register_named(
                kind="simulation",
                item=simulation,
                by_name=self._simulations_by_name,
                by_hash=self._simulation_hashes,
            )
            for simulation in simulations
        )

    def register_methods(self, methods: Iterable[MethodSpec]) -> tuple[MethodSpec, ...]:
        registered: list[MethodSpec] = []
        for method in methods:
            method_hash = self._item_hash(method)
            existing = self._methods_by_hash.get(method_hash)
            if existing is None:
                self._methods_by_hash[method_hash] = method
                existing = method
            registered.append(existing)
        return tuple(registered)

    def register_batches(self, batches: Iterable[BatchSpec]) -> tuple[BatchSpec, ...]:
        return tuple(
            self._register_named(
                kind="batch",
                item=batch,
                by_name=self._batches_by_name,
                by_hash=self._batch_hashes,
            )
            for batch in batches
        )

