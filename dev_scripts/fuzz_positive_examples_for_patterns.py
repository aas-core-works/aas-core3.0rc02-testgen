"""Generate strings matching the patterns in ``pattern_examples``."""

import warnings

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
        strategy = hypothesis.strategies.from_regex(pattern, fullmatch=True)

        print(f"For {pattern}:")
        print("[")
        for _ in range(3):
            text = strategy.example()
            print(f"  {text!r},")

        print("]")
        print()


if __name__ == "__main__":
    main()
