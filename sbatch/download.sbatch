#!/usr/bin/env bash
#SBATCH --job-name=glm53-download
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --output=logs/download-%j.out
#SBATCH --error=logs/download-%j.err

set -euo pipefail
# Use the submission directory, not the Slurm spool copy of this script.
ROOT="$(realpath "${SLURM_SUBMIT_DIR:?submit this job from the repository root}")"
[[ -x "${ROOT}/scripts/download-model.sh" ]] || {
  echo "SLURM_SUBMIT_DIR is not the repository root: ${ROOT}" >&2
  exit 2
}
cd "${ROOT}"
exec "${ROOT}/scripts/download-model.sh"
