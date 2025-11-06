# For docker
cd Soft-Thinking
docker build -t soft-thinking:st-cu124-py311 .
# NVIDIA Container Toolkit is required
docker run --gpus all --ipc=host --rm -it \
  -v $PWD:/workspace \
  soft-thinking:st-cu124-py311 bash

