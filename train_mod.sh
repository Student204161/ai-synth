
#!/bin/sh
#BSUB -q gpua100
#BSUB -J model_train
#BSUB -W 3:00
#BSUB -n 8
#BSUB -R "rusage[mem=20GB]"
#BSUB -gpu "num=1"
#BSUB -B
#BSUB -N
#BSUB -u s204161@student.dtu.dk
#BSUB -o model_train.out
#BSUB -e model_train.err

source /work3/s204161/miniconda/bin/activate py311

module load cuda/12.1

cd /work3/s204161/adlcv-proj/ai-synth/

python3 src/train_model_vivit.py