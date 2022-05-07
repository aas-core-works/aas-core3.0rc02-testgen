"""Generate strings not matching the patterns in ``pattern_examples``."""

import re
import warnings
from typing import Set

import hypothesis
import hypothesis.errors
import hypothesis.strategies

from aas_core3_0_rc02_testgen.frozen_examples import pattern as frozen_examples_pattern

warnings.filterwarnings(
    "ignore", category=hypothesis.errors.NonInteractiveExampleWarning
)


def main() -> None:
    """Execute the main routine."""
    for pattern in frozen_examples_pattern.BY_PATTERN.keys():
        pattern_re = re.compile(pattern)
        strategy = hypothesis.strategies.text()

        print(f"For {pattern}:")
        print("[")

        observed = set()  # type: Set[str]

        count = 0
        for _ in range(1000):
            text = strategy.example()
            if text in observed:
                continue

            if pattern_re.match(text) is None:
                print(f"  ('negatively_fuzzed_{(count+1):02d}', {text!r}),")
                count += 1
                observed.add(text)

            if count == 10:
                break

        print("]")
        print()


if __name__ == "__main__":
    main()
