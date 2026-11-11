from dataclasses import dataclass

from app.providers.base import EvidenceProvider


@dataclass(frozen=True)
class SourceRoute:
    """Explainable selection of the configured evidence providers."""

    providers: tuple[EvidenceProvider, ...]
    source_names: tuple[str, ...]
    reason: str


class SourceRouter:
    """Deterministically routes supported medical queries to configured sources."""

    def route(self, providers: list[EvidenceProvider]) -> SourceRoute:
        selected = tuple(providers)
        return SourceRoute(
            providers=selected,
            source_names=tuple(provider.name for provider in selected),
            reason="Query all configured evidence providers for broad biomedical coverage.",
        )
