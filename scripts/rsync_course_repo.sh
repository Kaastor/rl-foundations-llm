#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/course_repo"

mkdir -p "${DEST}"

rsync -a --delete \
  --exclude ".git/" \
  --exclude "scripts/" \
  --exclude "course_repo/" \
  --exclude ".env" \
  --exclude "notes/" \
  --exclude "runs/" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  --exclude "*.pyc" \
  --exclude "**/*_sol.py" \
  --exclude "tests/test_kata_01.py" \
  --exclude "tests/test_selection_policy.py" \
  --exclude "tests/test_reward_regressions.py" \
  --exclude "docs/solution_walkthrough.md" \
  --exclude "docs/learn/Walkthrough.md" \
  --exclude "docs/learn/Walkthrough_PL.md" \
  --exclude "docs/papers/" \
  --exclude "data/datasets/math_dev_TAMPERED.jsonl" \
  --exclude "data/golden/golden_exploits_extra.jsonl" \
  --exclude "AGENTS.md" \
  --exclude "docs/Blueprint - concepts.md" \
  "${ROOT}/" "${DEST}/"

# Overwrite assignment files with templates (student-facing).
cp "${DEST}/course/assignments/kata_01_template.py" "${DEST}/course/assignments/kata_01.py"
cp "${DEST}/course/assignments/selection_policy_template.py" "${DEST}/course/assignments/selection_policy.py"
cp "${DEST}/course/assignments/two_step_mdp_demo_template.py" "${DEST}/course/assignments/two_step_mdp_demo.py"
cp "${DEST}/course/assignments/kl_regularized_choice_template.py" "${DEST}/course/assignments/kl_regularized_choice.py"
cp "${DEST}/course/assignments/hackable_scorer_demo_template.py" "${DEST}/course/assignments/hackable_scorer_demo.py"

# Overwrite tests with templates so pytest sees student scaffolds.
cp "${DEST}/tests/kata_01_template.py" "${DEST}/tests/test_kata_01.py"
cp "${DEST}/tests/selection_policy_template.py" "${DEST}/tests/test_selection_policy.py"
cp "${DEST}/tests/reward_regressions_template.py" "${DEST}/tests/test_reward_regressions.py"

# Remove template files to keep the student repo coherent.
rm -f "${DEST}/course/assignments/"*_template.py
rm -f "${DEST}/tests/"*_template.py

echo "Wrote student repo to: ${DEST}"
