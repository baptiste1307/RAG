import fire
import re
import sys
from src.cli import CLI


def _validate_raw_cli_args(argv: list[str]) -> None:
    """Reject ambiguous command-line flags before Python Fire runs commands."""
    glued_integer_flags = {
        "k": "--k 10 or --k=10",
        "max_chunk_size": "--max_chunk_size 2000 or --max_chunk_size=2000",
        "max_context_length": (
            "--max_context_length 2000 or --max_context_length=2000"
        ),
    }
    for arg in argv[1:]:
        for flag_name, expected_usage in glued_integer_flags.items():
            if re.fullmatch(rf"--{flag_name}\d+", arg):
                raise ValueError(
                    f"Invalid argument '{arg}'. Use {expected_usage}; "
                    f"do not glue the value to --{flag_name}."
                )


def main() -> None:
    """Run the CLI and print short errors instead of long tracebacks."""
    try:
        _validate_raw_cli_args(sys.argv)
        fire.Fire(CLI)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
