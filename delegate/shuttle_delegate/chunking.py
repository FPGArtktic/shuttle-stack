# SPDX-License-Identifier: GPL-3.0-only
"""Cutting a file into pieces that fit a context.

The cuts are made at the largest boundary that works: blank lines
first, then single lines, and only then mid-line, so that a piece sent
to a model is as close to a whole thought as the size allows.
"""

from __future__ import annotations

PARAGRAPH = "\n\n"


def _hard_split(piece: str, limit: int) -> list[str]:
    return [piece[i : i + limit] for i in range(0, len(piece), limit)]


def _split_lines(piece: str, limit: int) -> list[str]:
    out: list[str] = []
    for line in piece.split("\n"):
        out.extend(_hard_split(line, limit) if len(line) > limit else [line])
    return out


def pieces(text: str, limit: int) -> list[str]:
    """Break the text down until no piece is longer than the limit."""
    out: list[str] = []
    for paragraph in text.split(PARAGRAPH):
        if len(paragraph) <= limit:
            out.append(paragraph)
        else:
            out.extend(_split_lines(paragraph, limit))
    return [piece for piece in out if piece.strip()]


def chunk(text: str, limit: int) -> list[str]:
    """Pack the text into as few chunks of at most `limit` characters.

    An empty text yields no chunks, so a caller never sends a prompt
    with nothing in it.

    The packing is not byte-for-byte faithful and is not meant to be:
    a paragraph too long for the limit is split on its lines and the
    pieces are rejoined with a blank line between them, and lines that
    held only whitespace are dropped. What the model reads is
    therefore the text's content rather than its exact shape. Nothing
    downstream depends on the shape: the grounding check collapses
    whitespace on both sides before comparing.
    """
    if limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    chunks: list[str] = []
    current = ""
    for piece in pieces(text, limit):
        if not current:
            current = piece
        elif len(current) + len(PARAGRAPH) + len(piece) <= limit:
            current += PARAGRAPH + piece
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks
