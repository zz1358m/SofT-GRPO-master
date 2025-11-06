conda create -n st python=3.11 -y && conda activate st
pip install --upgrade pip
pip install torch transformers accelerate jsonlines math_verify openai torch_memory_saver
pip install "datasets<4.0.0"
pip install flash_attn --no-build-isolation # may take more time (20min). try `pip install flash_attn==2.7.3 --no-build-isolation` if find undefined symbol bug

# install SGlang (0.4.6.post1) tailored for soft thinking
cd sglang_soft_thinking_pkg
pip install -e "python[all]"
cd ..

# install LiveCodeBench (0.1.0) if you want to evaluate on LiveCodeBench
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
mv LiveCodeBench LiveCodeBench_pkg
cd LiveCodeBench_pkg
pip install -e . --no-deps
cd ..


# For docker
cd Soft-Thinking
docker build -t soft-thinking:st-cu124-py311 .
# NVIDIA Container Toolkit is required
docker run --gpus all --rm -it \
  -v $PWD:/workspace \
  soft-thinking:st-cu124-py311 bash

# check version
# python -V
# python -c "import torch; print(torch.__version__, torch.version.cuda)"
# python -c "import sglang; print(sglang.__version__)"