#!/bin/bash

# Set error handling
set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check last command status
check_status() {
    if [ $? -eq 0 ]; then
        log "SUCCESS: $1"
    else
        log "ERROR: $1"
        exit 1
    fi
}

# Create installation directory
log "Starting installation..."

# 1. Install Anaconda
log "Downloading Anaconda..."
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
check_status "Anaconda download"

log "Installing Anaconda..."
bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p /anaconda3
check_status "Anaconda installation"

# 2. Setup conda
log "Setting up conda..."
export PATH="/anaconda3/bin:$PATH"
conda create -n vattn python=3.10 -y
check_status "Conda environment creation"

# Source conda for the script
source /anaconda3/etc/profile.d/conda.sh
conda activate vattn
check_status "Conda environment activation"

# 3. Clone vattention
log "Cloning vattention repository..."
git clone https://github.com/xutingl/vattention-ee.git
check_status "Repository cloning"

# 4. Install PyTorch
log "Installing PyTorch..."
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
check_status "PyTorch installation"

# 5. Install sarathi-lean requirements
log "Installing sarathi-lean requirements..."
cd vattention/sarathi-lean
pip install -r requirements.txt
pip install git+https://github.com/flashinfer-ai/flashinfer.git # pip install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.3/
python setup.py develop
check_status "sarathi-lean installation"

# 6. Install libtorch
log "Downloading and installing libtorch..."
cd /
wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.3.0%2Bcu121.zip
unzip libtorch-shared-with-deps-2.3.0+cu121.zip
check_status "libtorch installation"

# 7. Install vattention
log "Installing vattention..."
export LIBTORCH_PATH=/workspace/libtorch
cd /vattention/vattention
python setup.py install
check_status "vattention installation"

# Add conda initialization to .bashrc
log "Setting up conda initialization..."
echo ". /home/xinyi/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
echo "conda activate vattn" >> ~/.bashrc
check_status "Conda initialization setup"

export LD_LIBRARY_PATH=/anaconda3/envs/vattn/lib/python3.10/site-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/libtorch/lib:$LD_LIBRARY_PATH

log "Installation completed successfully!"