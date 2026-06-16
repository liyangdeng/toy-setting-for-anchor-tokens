#!/usr/bin/env python3
"""Helpers for extracting quoted terminals and POS labels from grammar templates."""

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import TypedDict


type Grammar = dict[str, str]
type TerminalIndex = dict[str, dict[str, set[str]]]


class InlineTerminal(TypedDict):
    terminal: str
    lhs: str
    relation: str
    production: str
    line_number: str


def load_grammar(path: Path) -> Grammar:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "grammar" for target in node.targets):
            grammar = ast.literal_eval(node.value)
            if not isinstance(grammar, dict):
                raise ValueError(f"`grammar` in {path} is not a dict")
            return grammar
    raise ValueError(f"No top-level `grammar = ...` assignment found in {path}")


def strip_probability(rhs_alt: str) -> str:
    return re.sub(r"\s*\[[^\]]+\]\s*$", "", rhs_alt).strip()


def is_single_terminal_production(production: str, quoted_terms: list[str]) -> bool:
    if len(quoted_terms) != 1:
        return False

    without_quotes = re.sub(r"'[^']+'", " ", production)
    without_probabilities = re.sub(r"\[[^\]]+\]", " ", without_quotes)
    leftover_tokens = without_probabilities.split()
    return not any(re.search(r"[A-Z_]", token) for token in leftover_tokens)


def parse_grammar(grammar: Grammar) -> tuple[TerminalIndex, list[InlineTerminal]]:
    terminals: TerminalIndex = defaultdict(lambda: defaultdict(set))
    inline_terminals: list[InlineTerminal] = []

    for relation, grammar_text in grammar.items():
        for line_number, raw_line in enumerate(grammar_text.splitlines(), start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line or "->" not in line:
                continue

            lhs, rhs = line.split("->", 1)
            lhs = lhs.strip()

            for alt in re.split(r"\s*\|\s*", rhs):
                production = strip_probability(alt)
                quoted_terms = re.findall(r"'([^']+)'", production)
                symbols = re.sub(r"'[^']+'", "TERM", production).split()

                if symbols == ["TERM"] or is_single_terminal_production(production, quoted_terms):
                    terminals[quoted_terms[0]][lhs].add(str(relation))
                    continue

                for term in quoted_terms:
                    inline_terminals.append(
                        {
                            "terminal": term,
                            "lhs": lhs,
                            "relation": str(relation),
                            "production": production,
                            "line_number": str(line_number),
                        }
                    )

    return terminals, inline_terminals


def base_pos(pos: str) -> str:
    for prefix in ("ADJP", "AUX", "CONJ", "PP", "V"):
        if pos.startswith(prefix):
            return prefix
    return pos
