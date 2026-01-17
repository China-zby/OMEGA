# OMEGA

We propose **OMEGA**, a framework for scalable OMQ (Object-centric Multi-modal Query) optimization with lightweight preprocessing overhead. This repository contains the implementation used in our experiments.

---

## Dataset

The dataset consists of the following components:

| Dataset | Description | Download URL |
|---------|-------------|--------------|
| Seattle | Traffic video dataset | https://web.seattle.gov/Travelers |
| YouTube-8M | Large-scale video dataset | https://research.google.com/youtube8m/download.html |

Download and extract the datasets following the URLs above before running experiments.

---

## Installation

### Prerequisites

- Python 3.8
- Conda (recommended)

### Setup Environment

```bash
# Create and activate conda environment
conda create -n omega python=3.8
conda activate omega

# Navigate to project directory
cd /path/to/omega/

# Install dependencies
pip install -r requirements.txt

## Run Experiments

Our method follows a two-stage pipeline: **Offline Training** and **Online Query**.

---

### Stage 1: Offline Training

Train the reinforcement learning model using the provided script.

#### Quick Start

```bash
bash train.sh

Custom Training
You can also run the training script directly with custom parameters:

python mix_reinforce_dpn_new.py \
    --input_epoch <epoch_number> \
    --query_path '<path_to_query_datasets>' \
    --video_path '<path_to_video>' \
    --cache_path '<path_to_cache>'

#### Training Parameters

- `--input_epoch`: Training iteration index (Example: `0`, `1`, ..., `9`)
- `--query_path`: Path to query dataset (Example: `/data/query_datasets/query_total_car`)
- `--video_path`: Path to input video file (Example: `/data/youtube-total.mp4`)
- `--cache_path`: Path to cache file .joblib (Example: `/data/cache/youtube_cache.joblib`)

#### Training Outputs

After training, the following files will be generated:

- `policy_model.h5`: Trained policy network weights
- `q_table.npy`: Q-table for reinforcement learning
- `checkpoint/`: Model checkpoints during training

---

### Stage 2: Online Query

After training, run the query process using the trained model:

```bash
python ./fast-reid/demo/baseline.py

Query Parameters
--test_baseline (bool, default: true): Set true for baseline method; false to enable RL-enhanced query
Example Commands
Run with RL-enhanced method (our approach):

python ./fast-reid/demo/baseline.py --test_baseline false
Run baseline comparison:

python ./fast-reid/demo/baseline.py --test_baseline true