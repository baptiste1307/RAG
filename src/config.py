import json
import sys
from pathlib import Path
from typing import Any
from dataclasses import dataclass


def load_json(path: Path) -> Any:
    """Load and decode a JSON file from disk.

    Many commands need to read datasets, chunks, or generated result files.
    This helper centralizes that behavior. If the file does not exist,
    it prints
    a clear error message and exits instead of letting the rest of the pipeline
    fail with a confusing traceback.
    """
    if not path.exists() or not path.is_file():
        print(f"Error: JSON file not found: {path}", file=sys.stderr)
        sys.exit(1)
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            # le fichier n'est pas du JSON valide
            raise Exception(f"Error: invalid JSON in {path}: {e}")
        except OSError as e:
            # le système npp ouvrir/lire/écrire ce fichier, même s’il existe.
            raise Exception(f"Error: cannot read file {path}: {e}")


@dataclass
class RagConfig:
    """Store the important paths and constants used by the project.

    Instead of repeating paths everywhere, the code imports `global_state` and
    reads these attributes. This makes it easy to find where chunks, indexes,
    datasets, and the raw vLLM repository are expected to live.
    """

    save_chunks_path = Path("./data/processed/chunks/chunks.json")
    save_index_path = Path("./data/processed/bm25_index")
    dataset_path = Path(
        "./datasets_public/public/UnansweredQuestions/"
        "dataset_code_public.json"
    )
    repo_path = Path("./data/raw/vllm-0.10.1")
    overlap = 200


global_state = RagConfig()
