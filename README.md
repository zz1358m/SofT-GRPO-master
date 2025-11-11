<!-- <p align="center" width="100%">
<img src="./docs/static/images/logo_resize.png"  width="80%">
</p> -->

<div align="center">
    <h1 align="center"> SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization
    </h1>
</div>

<p align="center">
  <img src="assets/mainprocess.png">
</p>



- **Authors**: [Zhi Zheng](https://zz1358m.github.io/zhizheng.github.io/), [Wee Sun Lee](https://scholar.google.com/citations?user=8PCrLgwAAAAJ&hl=en)
- **Institutes**: School of Computing, National University of Singapore, Singapore; 
- **Resources**: [📖[Paper](https://arxiv.org/abs/2511.06411)] [[🏠Twitter](https://x.com/zhengzhi20/status/1988071729789628731)] [[🤗Huggingface](https://huggingface.co/zz1358m/SofT-GRPO-master)]



## 📧 Welcome to feedback

We greatly appreciate your feedback and questions regarding the current status of this work. 

Please feel free to contact Zhi Zheng at [zhi.zheng@u.nus.edu](zhi.zheng@u.nus.edu)


## 💡 Highlights

- 🔥 **The First Powerful RLVR Algorithm for Soft-Thinking Reasoning:** We introduce **SofT-GRPO**, a novel and powerful policy optimization algorithm designed for reinforcing the soft-thinking reasoning paradigm in LLMs. 

- ⚙️ **Gumbel-Softmax Noise in Rollout:** It integrates the Gumbel-Softmax technique into the group rollout process, actively obtaining diverse but valid soft-thinking reasoning paths.

- ⚙️ **Gumbel Reparameterization:** We propose an innovative gradient estimation approach via Gumbel reparameterization, enabling precise attribution of improvements to the LLM’s output probability distributions in policy optimization. 

- 📝 **Comprehensive Experiments and High Effectiveness:** We conduct comprehensive experiments across LLMs of 1.5B–7B parameters on five benchmarks, demonstrating that SofT-GRPO consistently outperforms the discrete-token GRPO baselines, especially at higher sample rates (Pass@16 and Pass@32). SofT-GRPO can also improve the out-of-Domain generalization ability of LLMs.
 
- 🔥 **Showing the Prospects of Soft-Thinking:** Can Soft-Thinking be the Answer for Better Effectiveness?

## 📜 News

**[2025/9/24]** [Code](https://github.com/zz1358m/SofT-GRPO-master), [Weight](https://huggingface.co/zz1358m/SofT-GRPO-master), and [Paper](https://arxiv.org/abs/2511.06411) are released!

## 👨‍💻 Todo

- [x] SGLang & verl Code Modification (e.g., activate the overlap for efficiency).


## 🛠️ Usage

### 1. Clone the repository
```bash
git clone https://github.com/zz1358m/SofT-GRPO-master
cd SofT-GRPO-master
```

### 2. Install dependencies
##### Option 1: For inference only,
```bash
conda create -n soft_grpo python=3.11.13 -y && conda activate soft_grpo
pip install pip==25.2
pip install torch==2.6.0 transformers==4.51.1 tensorboard==2.20.0 sgl_kernel==0.1.1 accelerate==1.10.1 torch_memory_saver==0.0.8 uvloop==0.21.0 jsonlines math_verify openai
pip install flash_attn==2.7.3  --no-build-isolation # may take more time (20min). try `pip install flash_attn==2.7.3 --no-build-isolation` if find undefined symbol bug, or try downloading from its official github.

cd Soft-Thinking+noise+loss-main/sglang_soft_thinking_pkg
pip install -e "python[all]"
cd ../..
```

##### Option 2: For inference & SofT-GRPO fine-tuning,

building the verl-0.4.x after doing the Option1.
```bash
cd verl-0.4.x
pip3 install -e .
cd ..
```

or trying to install requirements. (not recommended)
```bash
pip install -r requirements.txt
```


---

### 3. Evaluating SofT-GRPO fine-tuned LLMs with soft-thinking pattern

#### Step 1: Download the SofT-GRPO, GRPO, weights from [[🤗Huggingface](https://huggingface.co/zz1358m/SofT-GRPO-master)]

#### Step 2: Evaluating GRPO under the discrete-token CoT pattern.
```bash
./Soft-Thinking+noise+loss-main/run_sample_discrete-token_grpo.sh
```

#### Step 3: Evaluating GRPO under the soft-thinking reasoning pattern.
```bash
./Soft-Thinking+noise+loss-main/run_sample_gumbel_grpo.sh
```

#### Step 3: Evaluating SofT-GRPO under the soft-thinking reasoning pattern.
```bash
./Soft-Thinking+noise+loss-main/run_sample_gumbel.sh
```


---

### 4. Training with SofT-GRPO

#### Option 1: Train the SofT-GRPO on DeepSeek-R1-Distill-Qwen-1.5B
```bash
./SofT-GRPO-deepscaler-8k.sh # change the LLM path, dataset path accordingly
```

#### Option 2: Train the SofT-GRPO on DeepSeek-R1-Distill-Qwen-7B
```bash
./SofT-GRPO-deepscaler-8k-qwen7.sh # change the LLM path, dataset path accordingly
```


#### Option 3: Train the SofT-GRPO on Llama-3.2-3B-Instruct
```bash
./SofT-GRPO-deepscaler-8k-llama3.sh # change the LLM path, dataset path accordingly
```




## ✒️ Citation

If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝

```bibtex
@misc{zheng2025softgrposurpassingdiscretetokenllm,
      title={SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization}, 
      author={Zhi Zheng and Wee Sun Lee},
      year={2025},
      eprint={2511.06411},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2511.06411}, 
}
```

## ❤️ Acknowledgments

- [Soft-Thinking](https://github.com/eric-ai-lab/Soft-Thinking): The codebase we built upon. Thanks for their wonderful work.
- [verl-0.4.x](https://github.com/volcengine/verl/tree/v0.4.x): Our work is based on this codebase as well.
- [SIM-CoT](https://github.com/InternLM/SIM-CoT): We use their template for README.md!
- [Yu Gu](https://github.com/kuangrepi): Undergraduate student from Nanjing University, volunteer for helping in code re-organization!
