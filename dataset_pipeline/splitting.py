from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .postprocessing import LabelCandidate
else:  # pragma: no cover
    LabelCandidate = Any


@dataclass
class SessionGroup:
    frames: List[str] = field(default_factory=list)
    positives: int = 0


class DatasetSplitter:
    """Creates train/val/test splits grouped by session."""

    def __init__(
        self,
        splits_root: str | Path,
        strategy: str,
        proportions: Dict[str, float],
        min_images_per_split: int = 50,
        seed: int = 42,
    ) -> None:
        self.splits_root = Path(splits_root)
        self.strategy = strategy
        self.proportions = proportions
        self.min_images_per_split = min_images_per_split
        self.random = random.Random(seed)
        self.primary_split = (
            max(proportions, key=proportions.get) if proportions else None
        )

    def split(self, records: Iterable[LabelCandidate], dataset_name: str) -> Dict[str, List[str]]:
        groups = self._group_records(records)
        ordered_groups: List[Tuple[str, SessionGroup]] = list(groups.items())
        self.random.shuffle(ordered_groups)

        split_sessions: Dict[str, List[Tuple[str, SessionGroup]]] = {
            key: [] for key in self.proportions
        }
        split_counts = {key: 0 for key in self.proportions}
        split_positive_sessions = {key: 0 for key in self.proportions}

        split_names = list(self.proportions.keys())
        for session_id, group in ordered_groups:
            target_split = self._select_split(split_names, split_counts, split_positive_sessions, group)
            split_sessions[target_split].append((session_id, group))
            split_counts[target_split] += len(group.frames)
            if group.positives > 0:
                split_positive_sessions[target_split] += 1

        self._ensure_minimums(split_sessions, split_counts, split_positive_sessions)
        self._ensure_positive_coverage(split_sessions, split_counts, split_positive_sessions)
        self._ensure_minimums(
            split_sessions,
            split_counts,
            split_positive_sessions,
            respect_proportions=False,
        )

        split_buckets: Dict[str, List[str]] = {key: [] for key in split_sessions}
        for split_name, sessions in split_sessions.items():
            for _, group in sessions:
                split_buckets[split_name].extend(group.frames)

        self._write_manifest(dataset_name, split_buckets)
        return split_buckets

    def _group_records(self, records: Iterable[LabelCandidate]) -> Dict[str, SessionGroup]:
        groups: Dict[str, SessionGroup] = {}
        for record in records:
            if self.strategy == "group_by_session":
                key = record.frame.session_id
            else:
                key = record.frame.session_id
            group = groups.setdefault(key, SessionGroup())
            group.frames.append(record.frame.frame_path)
            if record.detections:
                group.positives += 1
        return groups

    def _ensure_minimums(
        self,
        split_sessions: Dict[str, List[Tuple[str, SessionGroup]]],
        split_counts: Dict[str, int],
        split_positive_sessions: Dict[str, int],
        respect_proportions: bool = True,
    ) -> None:
        total_images = sum(split_counts.values())
        if total_images == 0:
            return
        for split_name, proportion in self.proportions.items():
            if respect_proportions:
                expected = max(int(total_images * proportion), self.min_images_per_split)
            else:
                expected = self.min_images_per_split
            expected = min(expected, total_images)
            if expected == 0:
                continue
            while split_counts[split_name] < expected:
                donor = self._select_donor(split_counts, exclude=split_name)
                if donor is None:
                    break
                session = self._pop_session(split_sessions[donor], prefer_background=True)
                if session is None:
                    break
                session_id, group = session
                split_sessions[split_name].append((session_id, group))
                split_counts[split_name] += len(group.frames)
                split_counts[donor] -= len(group.frames)
                if group.positives > 0:
                    split_positive_sessions[split_name] += 1
                    split_positive_sessions[donor] -= 1

    def _load_ratio(self, counts: Dict[str, int], split: str) -> float:
        proportion = self.proportions.get(split, 0.0)
        if proportion <= 0:
            return float("inf") if counts[split] > 0 else 0.0
        return counts[split] / proportion

    def _select_split(
        self,
        split_names: List[str],
        split_counts: Dict[str, int],
        split_positive_sessions: Dict[str, int],
        group: SessionGroup,
    ) -> str:
        if group.positives > 0:
            if self.primary_split is None:
                candidates = split_names
            else:
                missing_non_primary = [
                    name
                    for name in split_names
                    if name != self.primary_split and split_positive_sessions[name] == 0
                ]
                if missing_non_primary:
                    candidates = missing_non_primary
                else:
                    candidates = [self.primary_split]
        else:
            candidates = split_names
        return min(candidates, key=lambda split: self._load_ratio(split_counts, split))

    def _select_donor(self, counts: Dict[str, int], exclude: str) -> str | None:
        candidates = [name for name in counts if name != exclude and counts[name] > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda split: self._load_ratio(counts, split))

    def _pop_session(
        self,
        sessions: List[Tuple[str, SessionGroup]],
        require_positive: bool = False,
        prefer_background: bool = False,
    ) -> Tuple[str, SessionGroup] | None:
        if prefer_background:
            for idx in range(len(sessions) - 1, -1, -1):
                if sessions[idx][1].positives == 0:
                    return sessions.pop(idx)
        for idx in range(len(sessions) - 1, -1, -1):
            _, group = sessions[idx]
            if require_positive and group.positives == 0:
                continue
            return sessions.pop(idx)
        return None

    def _ensure_positive_coverage(
        self,
        split_sessions: Dict[str, List[Tuple[str, SessionGroup]]],
        split_counts: Dict[str, int],
        split_positive_sessions: Dict[str, int],
    ) -> None:
        splits_missing = [name for name, count in split_positive_sessions.items() if count == 0]
        for split_name in splits_missing:
            donor = self._select_positive_donor(split_positive_sessions, exclude=split_name)
            if donor is None:
                break
            session = self._pop_session(split_sessions[donor], require_positive=True)
            if session is None:
                continue
            session_id, group = session
            split_sessions[split_name].append((session_id, group))
            split_counts[split_name] += len(group.frames)
            split_counts[donor] -= len(group.frames)
            split_positive_sessions[split_name] += 1
            split_positive_sessions[donor] -= 1

    def _select_positive_donor(
        self,
        split_positive_sessions: Dict[str, int],
        exclude: str,
    ) -> str | None:
        donors = [
            name
            for name, positives in split_positive_sessions.items()
            if name != exclude and positives > 1
        ]
        if not donors:
            donors = [
                name
                for name, positives in split_positive_sessions.items()
                if name != exclude and positives > 0
            ]
        if not donors:
            return None
        return max(donors, key=lambda name: split_positive_sessions[name])

    def _write_manifest(self, dataset_name: str, splits: Dict[str, List[str]]) -> None:
        self.splits_root.mkdir(parents=True, exist_ok=True)
        path = self.splits_root / f"{dataset_name}_splits.json"
        path.write_text(json.dumps(splits, indent=2))
