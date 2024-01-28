This is a group project developed by a team of four individuals.

## Research Question:
Does spatial resolution (i.e. meters per pixel) matter for self-supervised learning methods for satellite imagery? 

## Dataset
National Agriculture Imagery Program (NAIP)

Metadata:
- 405,000 image patches (80/20 Train/Test split)
- Train/Test split from disjoint tiles
- Image patches sampled from a wide spectrum of landscapes in the state of California
- 6 landcover/landuse classes: barren land, trees, grassland, roads, buildings and water bodies (specifications follow National Land Cover Data (NLCD) algorithm)
- 28 X 28 non-overlapping image patches
- 1 landcover/landuse class per image patch
- 4 bands: red, green, blue and Near Infrared (NIR) per image patch
- 1-m ground sample distance (GSD) with a horizontal accuracy that lies within six meters of photo-identifiable ground control points

Dataset Credit: Saikat Basu, Sangram Ganguly, Supratik Mukhopadhyay, Robert Dibiano, Manohar Karki and Ramakrishna Nemani, DeepSat - A Learning framework for Satellite Imagery, ACM SIGSPATIAL 2015 (https://csc.lsu.edu/~saikat/deepsat/)

## Environment Setup

After logging into NYU VPN, 

```
ssh [netid]@greene.hpc.nyu.edu      # Greene login node
srun --pty -c 2 --mem=10GB --gres=gpu:rtx8000:1 /bin/bash       # Requesting resources on Greene compute node
```
Then download and unzip the overlay filesystem, followed by downloading the singularity image to the same working directory.

```
cd /scratch/[netid]
scp greene-dtn:/scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz .
gunzip -vvv ./overlay-25GB-500K.ext3.gz
scp -rp greene-dtn:/scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif .
```

Now download Miniconda onto the overlay filesystem on Singularity 

```
singularity exec --bind /scratch --nv --overlay /scratch/[netid]/overlay-25GB-500K.ext3:rw /scratch/[netid]/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash
Singularity> cd /ext3/
Singularity> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
Singularity> bash ./Miniconda3-latest-Linux-x86_64.sh
PREFIX=/ext3/miniconda3       # If the installer asks you where to install, type in `/ext3/miniconda3`
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
```
Some admin to reroute cache locations for python libraries.

First, create folders in scratch: 
```
Singularity> export SCRATCH=/scratch/[netid]
Singularity> mkdir $SCRATCH/.cache
Singularity> mkdir $SCRATCH/.conda
```

Now remove all existing cache:

```
Singularity> cd ~
Singularity> rm -rfv .conda
Singularity> rm -rfv .cache
```
Now create symbolic links (symlinks) to scratch:
```
Singularity> ln -s $SCRATCH/.conda ./
Singularity> ln -s $SCRATCH/.cache ./
```
Now let's install a few libraries for this repository (CUDA Version: 12.2, Python 3.11.5, Torch 2.1.0+cu121):
```
## Make sure that your conda environment is activated (e.g. run conda activate)
Singularity> pip install torch
Singularity> pip install matplotlib
Singularity> pip install pandas
```
Now lets check:

Make sure pytorch is using GPUs!
```
(base) Singularity> python
>>> import torch
>>> torch.cuda.is_available()
True
>>>
```
Now let's set up a few more things for batch setting on HPC

```
Singularity> exit
export SCRATCH=/scratch/[netid]
cd $SCRATCH
wget https://nyu-cs2590.github.io/course-material/fall2023/section/sec03/gpu_job.slurm
```
We need to add the /ext3/env.sh script to your filesystem. 
```
singularity exec --bind /scratch --nv --overlay /scratch/[netid]/overlay-25GB-500K.ext3:rw /scratch/[netid]/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash
Singularity> wget https://nyu-cs2590.github.io/course-material/fall2023/section/sec03/env.sh -O /ext3/env.sh
Singularity> exit
```
Now lets update the gpu_job.slurm script for our purposes
```
#!/bin/bash
#SBATCH --job-name=capstone
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --requeue
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB


singularity exec --bind /scratch --nv --overlay /scratch/[netid]/overlay-25GB-500K.ext3:rw /scratch/[netid]/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python /scratch/[netid]/NAIP-resolution/encoder_test.py --epochs=10 --latent_dim=128 --batch_size=32 --subset_percentage=0.1 --loss=binary_crossentropy #change this up as needed
"
```
Next, let's clone this repository onto HPC on /scratch/[netid]
```
git clone git@github.com:Lindsay-Lab/NAIP-resolution.git
```

And finally, we can run our batch jobs using this command going forward 
```
sbatch gpu_job.slurm
```
Major acknowledgement to DS-GA 1011 NLP HPC Tutorial guide for most of this set up instructions.

## Train/Val Split

train_val_split.ipynb - Script for splitting training data into Train/Val (80/20) i.e. overall split for train/val/test is 64/16/20. This script should not need to run again. The files have been shared w the group on NYU HPC (/scratch/fc1132/data)

## Dataloader

dataloader.py - Script for generating multiple transformations of image data in various resolutions 

## Encoder/Autoencoder
**encoder.py** - houses model definitions for autoencoder, encoder, and decoder as well as functions for training and reconstruction plotting.

**simsiam.py** - houses model definitions for SimSiam (the projector, predictor and same encoder network as the autoencoder).

## Classifier
**classifier.py** - houses model definitions for Classifier and ConvClassifier as well as functions for training and reconstruction plotting.

## Running Experiments

The script `experiment.py` trains a self-supervised model to learn image encodings. The trained encoder is then used to train a classifier, which is then evaluated on a test set.

**Example:**
```
python experiment.py --epochs=10 --latent_shape flat --latent_dim 256  --batch_size=32 --subset_percentage=0.1 --encoder_image_shape=28 28 --classifier_image_shape=28 28 --dropout=0.3
```

**Args (important in bold):**

`--run_id (string)`: your run will be stored under this name (defaults to ascending ints)

`--epochs (int)`: number of epochs to train for

`--subset_percentage (float (0.0,1.0])`: subset size to train/evaluate on 

**`--latent_shape ("flat" or "conv")`** encoder output shape - either flat vector or feature maps for a convolutional classifier

`--latent_dim (int)`: depth of encoder output (3rd dim)

`--batch_size (int)`: batch size for train/test

**`--encoder_image_shape (int int)`**:  shape of images used to train the self-supervised model

**`--self_supervised_model ("autoencoder" or "simsiam")`**: self-supervised architecture to train encoder

**`--classifier_image_shape (int int)`**: shape of input images for classifier training

`--dropout (float)`: dropout rate for classifier

**Results:**

All the results are found under the `experiments` folder. The file `run_logs.csv` contains parameters and test accuracy for all runs, while model checkpoints and epochwise loss can be found in the models folder under its run_id. This structure is outlined below:
```
├── experiments
|   ├── Autoencoder
│   │   ├── run_logs.csv
│   |   ├── model
│   │   |   ├── {run_id}
│   │   |   │   ├──classifier_{epoch}.pth
│   │   |   │   ├──classifier_loss.csv
│   │   │   |   ├──{self_supervized_model}_{epoch}.pth
│   │   │   |   ├──{self_supervized_model}_loss.csv
|   ├── SimSiam
|   |   ├── run_logs.csv
|   │   ├── model
|   │   │   ├── {run_id}
|   │   │   │   ├──classifier_{epoch}.pth
|   │   │   │   ├──{self_supervized_model}_{epoch}.pth

```

**Retrieving All Results from Cluster:**

On Singularity: `scp -r experiments [netid]@greene.hpc.nyu.edu:[greene-destination-path]`

On greene: `scp -r [netid]@greene.hpc.nyu.edu:[greene-path]/experiments [local-destination-path]`

