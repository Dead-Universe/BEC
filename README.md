## ✨ Highlights

- **Building-MoE**: Transformer encoder–decoder with sparse MoE (Top-k) feed-forward layers  
- **CLRS** (Closed-Loop Routing Scheduler): stagewise temp/noise, entropy feedback, short pulses, expert revival  
- **LBL** (Load Balancing Loss) + **Huber** loss: robust + balanced experts  
- **Single forward pass** covers horizons {1, 6, 12, 24, 48, 96, 168} hours  
- **Engineering compliance**: CVRMSE / NMAE / NMBE with ASHRAE Guideline 14 (hourly) thresholds

## Installation

> Recommended: Python ≥ 3.10, PyTorch ≥ 2.2, CUDA ≥ 12.x

### Full installation

1) Create and activate an environment

```bash
# Example with conda
conda create -n building-moe python=3.11 -y
conda activate building-moe
```

2) Install the project (editable)
```bash
git clone https://github.com/NREL/BuildingsBench.git
cd BuildingsBench
pip install -e ".[benchmark]"
```

### Installing faiss-gpu

Due to a PyPI limitation, we have to install `faiss-gpu` (for KMeans) by directly downloading the wheel from [https://github.com/kyamagu/faiss-wheels/releases/](https://github.com/kyamagu/faiss-wheels/releases/).
Download the wheel for the python version you are using, then install it in your environment.

For example:

```bash
wget https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

pip install faiss_gpu-1.7.3-cp38-cp38-manylinux2014_x86_64.whl
```

## Download the datasets and metadata

The pretraining dataset and evaluation data is available for download [here](https://data.openei.org/submissions/5859) as tar files, or can be accessed via AWS S3 [here](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=buildings-bench). The benchmark datasets are < 1GB in size in total, but the pretraining data is ~110GB in size.

The pretraining data is divided into 4 compressed files
- `comstock_amy2018.tar.gz`
- `comstock_tmy3.tar.gz`
- `resstock_amy2018.tar.gz`
- `resstock_tmy3.tar.gz`

and one compressed file for the metadata
- `metadata.tar.gz`

The evaluation datasets are compressed into a single file
- `BuildingsBench.tar.gz`

Download all files to a folder on a storage device with at least 250GB of free space. Then, decompress all of the downloaded files. There will be a new subdirectory called `BuildingsBench`. **This is the data directory, which is different than the Github code repository, although both folders are named "BuildingsBench".**


### Setting environment variables

Set the environment variable `BUILDINGS_BENCH` to the path where the data directory `BuildingsBench` is located (created when untarring the data files). **This is not the path to this code repository.**

```bash
export BUILDINGS_BENCH=/path/to/BuildingsBench
```
