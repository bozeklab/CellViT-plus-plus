#!/bin/bash
#SBATCH --job-name=cellvit_bham_mppfix
#SBATCH --output=job_logs/cellvit_bham_mppfix_%j.out
#SBATCH --error=job_logs/cellvit_bham_mppfix_%j.err
#SBATCH --partition=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00

# Whole Birmingham cohort with the coordinate fix (postprocessing_cupy offsets without the
# downsampling factor, wsi_meta 0.12 um/px -> native level 1 at 0.2428 um/px). Same model and
# preprocessing as the original run in test.sh. Resumable: slides whose _cells.json and
# _cell_detection.json already exist in --outdir are skipped, so just resubmit if it times out.
# Submit from ~/CellViT-plus-plus.

mkdir -p job_logs
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate cellvit_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Node: $(hostname) | GPUs: $CUDA_VISIBLE_DEVICES"

python3 -u ./cellvit/detect_cells.py \
  --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
  --batch_size 4 \
  --outdir ./test-results/birmingham/mrxs_cohort_mpp_fix \
  process_dataset \
  --wsi_folder /projects/ag-bozek/data/birmingham_head_and_neck/PN \
  --wsi_extension mrxs \
  --wsi_properties '{"slide_mpp": 0.121398698884758, "magnification": 40}' \
  --preprocessing_config ./birmingham_lowmem_preprocessing.yaml
