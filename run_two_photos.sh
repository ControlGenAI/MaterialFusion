#!/bin/bash
#SBATCH --job-name=big_dataset            
#SBATCH --error=logs/big_dataset_w_o_gar-%j.err 
#SBATCH --output=logs/big_dataset_w_o_gar-%j.log   
#SBATCH --gpus=1                     
#SBATCH --cpus-per-task=8            
#SBATCH --time=0:35:00               
#SBATCH --constraint="[type_a|type_b|type_c]"  

module load Python/Anaconda_v03.2023

source deactivate
source activate garifullin_gar_zest
cd  ~/garifullin/MaterialFusion/
python3 two_photos_gen.py
echo "Task done!" 
