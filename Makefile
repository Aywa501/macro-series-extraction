# ===========================================================================
# Hansard Monetary-Idiom Pipeline — Makefile
#
# Usage
# -----
#   make all                 # full pipeline (01 → 06), CSV input mode
#   make all INPUT=xml       # full pipeline using the ZIP/XML bulk corpus
#   make 01                  # extract only
#   make 02                  # disambiguate only (requires 01)
#   make 03                  # embed only (requires 02)
#   make 04                  # score only (requires 03)
#   make 05                  # build index (requires 04)
#   make 06                  # regression (requires 05)
#   make compare-windows     # context-window experiment (requires 02)
#   make clean               # remove all derived outputs (NOT raw data)
#   make clean-interim       # remove only interim/ (candidates, observations)
#   make force-all INPUT=xml # force-rerun entire pipeline from scratch
#
# Variables
# ---------
#   INPUT        csv (default) | xml
#   CONTEXT_WIN  sentence context radius for stage 01 (default: 2, -1=full)
#   WORKERS      worker threads/processes where applicable
#   LOG_LEVEL    DEBUG | INFO | WARNING | ERROR (default: INFO)
# ===========================================================================

# ---- tuneable defaults ----
INPUT       ?= csv
CONTEXT_WIN ?= 2
WORKERS     ?= 5
LOG_LEVEL   ?= INFO

# ---- fixed paths ----
PYTHON      := python3
SRC         := src
DATA        := data
OUTPUTS     := outputs

CANDIDATES  := $(DATA)/interim/candidates.parquet
OBSERVATIONS:= $(DATA)/interim/observations.parquet
EMBEDDINGS  := $(DATA)/processed/embeddings/index.parquet
DRIFT       := $(DATA)/processed/drift.parquet
INDEX       := $(DATA)/processed/drift_index.parquet

# ---- sentinel stamps (touch files to record completion) ----
STAMP_DIR   := $(DATA)/interim/.stamps
$(shell mkdir -p $(STAMP_DIR))

STAMP_01    := $(STAMP_DIR)/01.done
STAMP_02    := $(STAMP_DIR)/02.done
STAMP_03    := $(STAMP_DIR)/03.done
STAMP_04    := $(STAMP_DIR)/04.done
STAMP_05    := $(STAMP_DIR)/05.done
STAMP_06    := $(STAMP_DIR)/06.done

# ===========================================================================
.PHONY: all 01 02 03 04 05 06 compare-windows substitution-test clean clean-interim force-all help

all: 06
	@echo ""
	@echo "✓  Pipeline complete."

# ---------------------------------------------------------------------------
# Stage 01 — Extract candidates
# ---------------------------------------------------------------------------
$(STAMP_01):
	@echo "==> Stage 01: extract (input=$(INPUT), context-window=$(CONTEXT_WIN)) …"
	$(PYTHON) $(SRC)/01_extract.py \
		--input-format $(INPUT) \
		--context-window $(CONTEXT_WIN) \
		--log-level $(LOG_LEVEL)
	@touch $@

01: $(STAMP_01)

# ---------------------------------------------------------------------------
# Stage 02 — LLM disambiguation
# ---------------------------------------------------------------------------
$(STAMP_02): $(STAMP_01)
	@echo "==> Stage 02: disambiguate (workers=$(WORKERS)) …"
	$(PYTHON) $(SRC)/02_disambiguate.py \
		--workers $(WORKERS) \
		--log-level $(LOG_LEVEL)
	@touch $@

02: $(STAMP_02)

# ---------------------------------------------------------------------------
# Stage 03 — Contextual embeddings
# ---------------------------------------------------------------------------
$(STAMP_03): $(STAMP_02)
	@echo "==> Stage 03: embed …"
	$(PYTHON) $(SRC)/03_embed.py \
		--log-level $(LOG_LEVEL)
	@touch $@

03: $(STAMP_03)

# ---------------------------------------------------------------------------
# Stage 04 — Semantic triviality scoring
# ---------------------------------------------------------------------------
$(STAMP_04): $(STAMP_03)
	@echo "==> Stage 04: score …"
	$(PYTHON) $(SRC)/04_score.py \
		--log-level $(LOG_LEVEL)
	@touch $@

04: $(STAMP_04)

# ---------------------------------------------------------------------------
# Stage 05 — Build index S_t
# ---------------------------------------------------------------------------
$(STAMP_05): $(STAMP_04)
	@echo "==> Stage 05: build index …"
	$(PYTHON) $(SRC)/05_index.py \
		--log-level $(LOG_LEVEL)
	@touch $@

05: $(STAMP_05)

# ---------------------------------------------------------------------------
# Stage 06 — Regression
# ---------------------------------------------------------------------------
$(STAMP_06): $(STAMP_05)
	@echo "==> Stage 06: regression …"
	$(PYTHON) $(SRC)/06_regression.py \
		--log-level $(LOG_LEVEL)
	@touch $@

06: $(STAMP_06)

# ---------------------------------------------------------------------------
# Context-window comparison experiment
# ---------------------------------------------------------------------------
compare-windows: $(STAMP_02)
	@echo "==> Context-window comparison (requires observations.parquet) …"
	$(PYTHON) $(SRC)/scripts/compare_context_windows.py \
		--log-level $(LOG_LEVEL)

# ---------------------------------------------------------------------------
# Substitution test (optional, requires observations.parquet + MacBERTh)
# ---------------------------------------------------------------------------
substitution-test: $(STAMP_02)
	@echo "==> Substitution test (MacBERTh MLM denomination masking) …"
	$(PYTHON) $(SRC)/scripts/substitution_test.py \
		--log-level $(LOG_LEVEL)

# ---------------------------------------------------------------------------
# Force-rerun entire pipeline from scratch
# ---------------------------------------------------------------------------
force-all: clean-interim
	@$(MAKE) all INPUT=$(INPUT) CONTEXT_WIN=$(CONTEXT_WIN) WORKERS=$(WORKERS)

# ---------------------------------------------------------------------------
# Clean targets
# ---------------------------------------------------------------------------
clean-interim:
	@echo "Removing interim data and stamps …"
	rm -rf $(DATA)/interim
	rm -rf $(DATA)/processed
	@echo "Done."

clean: clean-interim
	@echo "Removing output figures and tables …"
	rm -rf $(OUTPUTS)/figures
	rm -rf $(OUTPUTS)/tables
	@echo "Done."

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "  make all                  Full pipeline (CSV mode by default)"
	@echo "  make all INPUT=xml        Full pipeline (ZIP/XML bulk data)"
	@echo "  make 01 .. make 06        Individual stages"
	@echo "  make compare-windows      Context-window experiment"
	@echo "  make substitution-test    MacBERTh MLM denomination masking"
	@echo "  make force-all INPUT=xml  Force-rerun everything from scratch"
	@echo "  make clean                Remove all derived data and figures"
	@echo "  make clean-interim        Remove only data/interim + data/processed"
	@echo ""
	@echo "  Variables:"
	@echo "    INPUT=$(INPUT)  CONTEXT_WIN=$(CONTEXT_WIN)  WORKERS=$(WORKERS)  LOG_LEVEL=$(LOG_LEVEL)"
	@echo ""
