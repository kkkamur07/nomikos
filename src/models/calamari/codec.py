"""Deterministic character codec and CTC decoding for Calamari."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class CtcFrame:
    """One CTC time step after softmax: greedy class plus the top alternatives."""

    start: int
    end: int
    token_id: int
    character: str
    probability: float
    top: tuple[tuple[str, float], ...]


def _display_character(character: str) -> str:
    return "∅" if character == "" else character


@dataclass(frozen=True)
class CharacterCodec:
    """A blank-first, character-level CTC vocabulary."""

    charset: tuple[str, ...]

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> CharacterCodec:
        return cls(("", *sorted({character for text in texts for character in text})))

    def __post_init__(self) -> None:
        if (
            len(self.charset) < 2
            or self.charset[0] != ""
            or len(set(self.charset)) != len(self.charset)
        ):
            raise ValueError("A codec must have a unique blank entry at index zero.")

    @property
    def classes(self) -> int:
        return len(self.charset)

    def encode(self, text: str) -> Tensor:
        ids = {character: index for index, character in enumerate(self.charset)}
        try:
            return torch.tensor([ids[character] for character in text], dtype=torch.long)
        except KeyError as error:
            raise ValueError(f"Character {error.args[0]!r} is absent from the codec.") from error

    def decode_ctc(self, token_ids: Sequence[int]) -> str:
        output: list[str] = []
        previous = -1
        for token_id in token_ids:
            if token_id != 0 and token_id != previous:
                output.append(self.charset[token_id])
            previous = token_id
        return "".join(output)

    def decode_logits(self, logits: Tensor, lengths: Tensor) -> list[str]:
        predictions = logits.argmax(dim=-1).detach().cpu()
        return [
            self.decode_ctc(row[: int(length)].tolist())
            for row, length in zip(predictions, lengths.detach().cpu(), strict=True)
        ]

    def frame_distributions(
        self, logits: Tensor, length: int, *, top_k: int = 3
    ) -> list[CtcFrame]:
        """Softmax over classes at each valid time step, then merge repeated argmax runs."""
        if top_k < 1:
            raise ValueError("top_k must be at least one.")
        if length <= 0:
            return []
        probs = logits[:length].detach().float().softmax(dim=-1)
        greedy = probs.argmax(dim=-1)
        top_k = min(top_k, probs.shape[-1])
        top_prob, top_id = probs.topk(top_k, dim=-1)
        frames: list[CtcFrame] = []
        run_start = 0
        for index in range(int(length)):
            token_id = int(greedy[index])
            is_last = index + 1 == int(length)
            next_differs = not is_last and int(greedy[index + 1]) != token_id
            if not (is_last or next_differs):
                continue
            run_probs = probs[run_start : index + 1, token_id]
            weakest = run_start + int(run_probs.argmin())
            top = tuple(
                (_display_character(self.charset[int(class_id)]), float(probability))
                for class_id, probability in zip(top_id[weakest], top_prob[weakest], strict=True)
            )
            frames.append(
                CtcFrame(
                    start=run_start + 1,
                    end=index + 1,
                    token_id=token_id,
                    character=_display_character(self.charset[token_id]),
                    probability=float(run_probs.mean()),
                    top=top,
                )
            )
            run_start = index + 1
        return frames

    def format_frame_distributions(
        self,
        logits: Tensor,
        length: int,
        *,
        reference: str | None = None,
        top_k: int = 3,
    ) -> str:
        """Pretty-print greedy CTC path with the top softmax rivals at each run."""
        frames = self.frame_distributions(logits, length, top_k=top_k)
        decoded = self.decode_ctc(logits[:length].argmax(dim=-1).detach().cpu().tolist())
        lines = []
        if reference is not None:
            lines.append(f"GT:   {reference}")
        lines.append(f"PRED: {decoded}")
        lines.append(f"{'t':<10}{'argmax':<8}{'p':<8}{'also'}")
        for frame in frames:
            span = (
                str(frame.start)
                if frame.start == frame.end
                else f"{frame.start}-{frame.end}"
            )
            rivals = "  ".join(
                f"{character} {probability:.2f}"
                for character, probability in frame.top
                if character != frame.character
            )
            mark = "  < low" if frame.probability < 0.5 else ""
            lines.append(
                f"{span:<10}{frame.character:<8}{frame.probability:<8.2f}{rivals}{mark}"
            )
        return "\n".join(lines)
