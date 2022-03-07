#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=5:00
#SBATCH --account=def-kyi-ab

idr_root="$HOME/projects/def-kyi-ab/jross02/idr-dif"
cd "$SLURM_TMPDIR"
mkdir collection5
cd collection5
unzip "$idr_root"/data/idr/collection5.zip
if [ -L "$idr_root"/data/idr/collection5 ]; then
    rm "$idr_root"/data/idr/collection5
fi
ln -s "$SLURM_TMPDIR"/collection5 "$idr_root"/data/idr/collection5
"$idr_root"/venv/bin/python "$idr_root"/code/training/exp_runner.py --conf "$idr_root"/code/confs/srn_collection_fixed_cameras.conf --collection 5