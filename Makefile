PYTHON = python3 -m
RM = rm -rf
RUN = uv run
ARGS ?=

install:
	uv sync --extra dev

run:
# bytecode non généré (.pyc, __pycache__)
	PYTHONDONTWRITEBYTECODE=1 $(RUN) $(PYTHON) src $(ARGS)

debug:
	$(RUN) $(PYTHON) pdb -m src

test:

clean:
	$(RM) .venv
	$(RM) .pytest_cache .mypy_cache .ipynb_checkpoints
	$(RM) data/processed
# "-delete" pour suppr fichiers directement
	find . -name "*.pyc" -delete
# "-exec..." pour appeler rm car egg-info peut être un fichier OU un dossier
# "+" pour ne pas éxécuter rm à chaque fichier trouvé: regroupe tout en une seule commande
	find . -name "*.egg-info" -exec $(RM) {} + 

lint:
	$(RUN) $(PYTHON) flake8 .  \
		--exclude=.venv,venv,.git,__pycache__,.mypy_cache,.pytest_cache,data,moulinette_pkg \
		--max-line-length=80 \
		--extend-ignore=E203,W503
	$(RUN) $(PYTHON) mypy \
		--warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs . \
		--exclude '.venv|venv|data|moulinette_pkg'


lint-strict:
	$(PYTHON) flake8 .
	$(PYTHON) mypy --strict .

.PHONY: install run debug test clean lint lint-strict
