from __future__ import annotations

from typing import Iterable

from core import HASH_KEY, MethodSpec, SimulationSpec, dehydrate_hashed
from utils import BatchSpec, CollectionSpec


class ConfigRegistry:
    def __init__(self) -> None:
        self._simulations_by_name: dict[str, SimulationSpec] = {}
        self._simulation_hashes: dict[str, SimulationSpec] = {}
        self._methods_by_hash: dict[str, MethodSpec] = {}
        self._batches_by_name: dict[str, BatchSpec] = {}
        self._batch_hashes: dict[str, BatchSpec] = {}
        self._collections_by_name: dict[str, CollectionSpec] = {}
        self._collection_hashes: dict[str, CollectionSpec] = {}

    @property
    def simulations(self) -> tuple[SimulationSpec, ...]:
        return tuple(self._simulations_by_name.values())

    @property
    def methods(self) -> tuple[MethodSpec, ...]:
        return tuple(self._methods_by_hash.values())

    @property
    def batches(self) -> tuple[BatchSpec, ...]:
        return tuple(self._batches_by_name.values())

    @property
    def collections(self) -> tuple[CollectionSpec, ...]:
        return tuple(self._collections_by_name.values())

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

    def register_collection(
        self,
        *,
        name: str,
        methods: Iterable[MethodSpec],
        simulations: Iterable[SimulationSpec] = (),
        n_batches: int | None = None,
        replicates_per_batch: int | None = None,
        batches: Iterable[BatchSpec] | None = None,
        batch_builder=None,
    ) -> CollectionSpec:
        if batches is None:
            if n_batches is None or replicates_per_batch is None or batch_builder is None:
                raise ValueError(
                    "register_collection requires either explicit batches or a batch policy."
                )
            registered_simulations = self.register_simulations(simulations)
            batch_specs = tuple(
                batch
                for simulation in registered_simulations
                for batch in batch_builder(
                    simulation,
                    replicates_per_batch=replicates_per_batch,
                    n_batches=n_batches,
                )
            )
        else:
            if n_batches is not None or replicates_per_batch is not None:
                raise ValueError(
                    "register_collection cannot accept both explicit batches and a batch policy."
                )
            batch_specs = tuple(batches)
            self.register_simulations(batch.simulation_spec for batch in batch_specs)

        registered_methods = self.register_methods(methods)
        registered_batches = self.register_batches(batch_specs)
        collection = CollectionSpec(
            name=name,
            batches=registered_batches,
            method_specs=registered_methods,
        )
        return self._register_named(
            kind="collection",
            item=collection,
            by_name=self._collections_by_name,
            by_hash=self._collection_hashes,
        )

    def register_collection_union(
        self,
        *,
        name: str,
        collections: Iterable[str],
    ) -> CollectionSpec:
        source_names = tuple(collections)
        source_collections: list[CollectionSpec] = []
        for collection_name in source_names:
            collection = self._collections_by_name.get(collection_name)
            if collection is None:
                raise KeyError(f"Unknown collection {collection_name!r}.")
            source_collections.append(collection)

        batch_hashes: set[str] = set()
        union_batches: list[BatchSpec] = []
        method_hashes: set[str] = set()
        union_methods: list[MethodSpec] = []

        for collection in source_collections:
            for batch in collection.batches:
                batch_hash = self._item_hash(batch)
                if batch_hash not in batch_hashes:
                    batch_hashes.add(batch_hash)
                    union_batches.append(batch)
            for method in collection.method_specs:
                method_hash = self._item_hash(method)
                if method_hash not in method_hashes:
                    method_hashes.add(method_hash)
                    union_methods.append(method)

        return self.register_collection(
            name=name,
            batches=tuple(union_batches),
            methods=tuple(union_methods),
        )
