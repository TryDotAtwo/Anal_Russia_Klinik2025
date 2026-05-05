from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .models import Term, TermVariant
from .text import normalize_query


@dataclass
class _Node:
    children: dict[str, int] = field(default_factory=dict)
    fail: int = 0
    outputs: list[TermVariant] = field(default_factory=list)


class AhoCorasickMatcher:
    def __init__(self, terms: list[Term]) -> None:
        self.nodes: list[_Node] = [_Node()]
        for term in terms:
            for variant in term.variants:
                normalized = normalize_query(variant)
                if normalized:
                    self._add(
                        TermVariant(
                            term_id=term.term_id,
                            source=term.source,
                            canonical=term.canonical,
                            variant=variant,
                            normalized=normalized,
                            metadata=term.metadata,
                        )
                    )
        self._build_failures()

    def _add(self, variant: TermVariant) -> None:
        node_index = 0
        for char in variant.normalized:
            node = self.nodes[node_index]
            if char not in node.children:
                node.children[char] = len(self.nodes)
                self.nodes.append(_Node())
            node_index = node.children[char]
        self.nodes[node_index].outputs.append(variant)

    def _build_failures(self) -> None:
        queue: deque[int] = deque()
        for child_index in self.nodes[0].children.values():
            queue.append(child_index)
            self.nodes[child_index].fail = 0
        while queue:
            current_index = queue.popleft()
            current = self.nodes[current_index]
            for char, child_index in current.children.items():
                queue.append(child_index)
                fail_index = current.fail
                while fail_index and char not in self.nodes[fail_index].children:
                    fail_index = self.nodes[fail_index].fail
                self.nodes[child_index].fail = self.nodes[fail_index].children.get(char, 0)
                self.nodes[child_index].outputs.extend(self.nodes[self.nodes[child_index].fail].outputs)

    def finditer(self, normalized_text: str) -> list[tuple[int, int, TermVariant]]:
        results: list[tuple[int, int, TermVariant]] = []
        node_index = 0
        for position, char in enumerate(normalized_text):
            while node_index and char not in self.nodes[node_index].children:
                node_index = self.nodes[node_index].fail
            node_index = self.nodes[node_index].children.get(char, 0)
            for output in self.nodes[node_index].outputs:
                start = position - len(output.normalized) + 1
                results.append((start, position + 1, output))
        return results
