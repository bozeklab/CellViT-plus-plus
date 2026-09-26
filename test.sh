#!/bin/bash
#SBATCH --job-name=cellvit
#SBATCH --output=job_logs/job_%A_%a.out
#SBATCH --error=job_logs/job_%A_%a.err
#SBATCH --partition=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00

# Ensure log directory exists
mkdir -p job_logs

# Activate Conda environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate cellvit_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "=== Job Debug Info ==="
echo "Node: $(hostname)"
echo "Using Python from: $(which python)"
echo "GPUs Assigned by Slurm: $CUDA_VISIBLE_DEVICES"
echo "CPUs Allocated: $SLURM_CPUS_PER_TASK"
echo "======================"

# giessen
# python3 ./cellvit/detect_cells.py --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth --outdir ./test-results/giessen/folder process_dataset --wsi_folder /projects/ag-bozek/data/HNO/OPSCC_HNSC-TCGA-Giessen-Validation --wsi_extension ndpi 

#tcga_hnsc
# python3 ./cellvit/detect_cells.py --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth --outdir ./test-results/tcga_hnsc/folder process_dataset --wsi_folder /projects/ag-bozek/data/TCGA_HNSC --wsi_extension svs

# birmingham
python3 -u ./cellvit/detect_cells.py \
	--model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
	--batch_size 4 \
	--outdir ./test-results/birmingham/mrxs_cohort \
	process_dataset \
	--wsi_folder /projects/ag-bozek/data/birmingham_head_and_neck/PN \
	--wsi_extension mrxs \
	--wsi_properties '{"slide_mpp": 0.121398698884758, "magnification": 40}' \
	--preprocessing_config ./birmingham_lowmem_preprocessing.yaml

# cologne
# python3 ./cellvit/detect_cells.py --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth --outdir ./test-results/cologne/folder process_dataset --filelist ./opscc_hpv_filelist.csv
