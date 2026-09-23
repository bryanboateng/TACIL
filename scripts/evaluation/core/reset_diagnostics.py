"""Opt-in line timings for the native calls hidden inside SAPIEN builders."""

import json
import linecache
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter, process_time
from types import FrameType


class ResetLineProfiler:
    def __init__(self) -> None:
        from sapien.wrapper.actor_builder import ActorBuilder
        from sapien.wrapper.urdf_loader import URDFLoader

        from scripts.environments.peg_insertion.reusable_scene import (
            ReusablePegInsertionSideEnv,
        )

        self._codes = {
            ActorBuilder.build_physx_component.__code__,
            URDFLoader._build_link.__code__,
            ReusablePegInsertionSideEnv._load_objects.__code__,
        }
        self._frames: dict[int, tuple[int, float, float]] = {}
        self._timings: dict[tuple[str, int], list[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0]
        )

    def _trace(self, frame: FrameType, event: str, arg):
        if frame.f_code not in self._codes:
            return None
        if event not in ("line", "return"):
            return self._trace
        now, cpu = perf_counter(), process_time()
        previous = self._frames.pop(id(frame), None)
        if previous is not None:
            line, start, cpu_start = previous
            totals = self._timings[(frame.f_code.co_filename, line)]
            totals[0] += 1
            totals[1] += now - start
            totals[2] += cpu - cpu_start
        if event == "line":
            self._frames[id(frame)] = (frame.f_lineno, now, cpu)
        return self._trace

    def run(self, function, **kwargs):
        previous = sys.gettrace()
        try:
            sys.settrace(self._trace)
            return function(**kwargs)
        finally:
            sys.settrace(previous)
            self._frames.clear()

    def save(self, path: Path) -> None:
        rows = [
            {
                "file": filename,
                "line": line,
                "source": linecache.getline(filename, line).strip(),
                "calls": counts[0],
                "wall_seconds": counts[1],
                "cpu_seconds": counts[2],
            }
            for (filename, line), counts in self._timings.items()
        ]
        path.write_text(json.dumps(rows, indent=2))


def print_reset_line_profiles(directory: Path) -> None:
    combined = {}
    for path in directory.glob("*.json"):
        for row in json.loads(path.read_text()):
            key = (row["file"], row["line"])
            if key not in combined:
                combined[key] = row
            else:
                for field in ("calls", "wall_seconds", "cpu_seconds"):
                    combined[key][field] += row[field]
    print("\nSlow reset lines: inclusive timings summed across workers.")
    print(
        "Nested rows can overlap; do not sum them. CPU time includes process threads."
    )
    print(" wall(s)   CPU(s)   calls  location / source")
    for row in sorted(combined.values(), key=lambda r: r["wall_seconds"], reverse=True)[
        :20
    ]:
        print(
            f"{row['wall_seconds']:8.3f} {row['cpu_seconds']:8.3f} "
            f"{row['calls']:7.0f}  {Path(row['file']).name}:{row['line']}  {row['source']}"
        )
