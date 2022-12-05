#!/bin/bash
#SBATCH --mem=300GB
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres gpu:a100
#SBATCH --job-name=jupyter
# get tunneling info

XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="argo-comp2"
port=1402
# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}.princeton.edu
Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"
which python
which jupyter

jupyter-notebook --no-browser --port=${port} --ip=${node}


