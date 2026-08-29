"""Strict-denominator possession counter with retroactive adjustments.

Split out of the old video_analysis/possession.py. Used only by the local
dev script's summary/printout — the production service reimplements
equivalent counting independently in service/stats.py.
"""

from __future__ import annotations

from modules.possession.labels import POSSESS_LOOSE, POSSESS_OOF, POSSESS_TEAM_A, POSSESS_TEAM_B


class PossessionStats:
    """Counts in-play frames per outcome and exposes possession % over only
    the confirmed-team frames (loose / OOF are excluded).

    Provisional team credit accumulated during travel is corrected via
    `apply_adjustments` when the pass FSM resolves the event:
        ("flip_to", team_id, n)  → move n frames from the other team to team_id
        ("drop",    team_id, n)  → remove n frames from team_id, give them to OOF
    """

    def __init__(self) -> None:
        self.frame_counts: dict[str, int] = {
            POSSESS_TEAM_A: 0,
            POSSESS_TEAM_B: 0,
            POSSESS_LOOSE: 0,
            POSSESS_OOF:   0,
        }
        self.total = 0

    def update(self, label: str) -> None:
        self.frame_counts[label] = self.frame_counts.get(label, 0) + 1
        self.total += 1

    def apply_adjustments(self, adjustments: list[tuple[str, int, int]]) -> None:
        for kind, team_id, n in adjustments:
            # team_id 0 → team_a, 1 → team_b
            src_label = POSSESS_TEAM_A if team_id == 0 else POSSESS_TEAM_B
            other_label = POSSESS_TEAM_B if team_id == 0 else POSSESS_TEAM_A
            if kind == "flip_to":
                # We credited `other` provisionally; move n frames over to team_id.
                move = min(n, self.frame_counts[other_label])
                self.frame_counts[other_label] -= move
                self.frame_counts[src_label]   += move
            elif kind == "drop":
                # We credited team_id provisionally; reclassify n as OOF.
                move = min(n, self.frame_counts[src_label])
                self.frame_counts[src_label] -= move
                self.frame_counts[POSSESS_OOF] += move

    def percentages(self) -> tuple[float, float]:
        denom = self.frame_counts[POSSESS_TEAM_A] + self.frame_counts[POSSESS_TEAM_B]
        if denom == 0:
            return 0.0, 0.0
        return (
            100.0 * self.frame_counts[POSSESS_TEAM_A] / denom,
            100.0 * self.frame_counts[POSSESS_TEAM_B] / denom,
        )

    def summary(self) -> str:
        t0, t1 = self.percentages()
        d = max(1, self.total)
        return "\n".join([
            "--- Possession Summary (denominator excludes loose/OOF) ---",
            f"  Team A: {t0:.1f}%",
            f"  Team B: {t1:.1f}%",
            f"  Loose : {100*self.frame_counts[POSSESS_LOOSE]/d:.1f}%",
            f"  OOF   : {100*self.frame_counts[POSSESS_OOF]/d:.1f}%",
        ])
