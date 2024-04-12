
# Mitigating Multilingual Hallucination in Large Vision-Language Models
This is the official repo for Multilingual Hallucination Removal, a simple and effective method for mitigating multilingual hallucinations in LVLMs.

<div style='display:flex; gap: 0.25rem; '>
<a href='LICENCE'><img src='https://img.shields.io/badge/License-Apache 2.0-g.svg'></a>
</div>


## 🎯 Overview
![MHR](figs/main.jpg)
- We Multilingual Hallucination Removal, **a simple and training-free** method that contrasts output distributions derived from original and distorted visual inputs.
- The new **contrastive probability distribution** for decoding is formulated as follows:
```math
p_{vcd}(y \mid v, v', x) = softmax[ (1+\alpha)\times logit_\theta (y \mid v, x) - \alpha \times logit_\theta(y \mid v', x)],
```
- The proposed VCD effectively reduces the over-reliance on **statistical bias** and **unimodal priors**, two essential causes of object hallucinations.


## 🕹️ Usage
### Environment Setup
```bash
conda create -yn vcd python=3.9
conda activate vcd
cd VCD
pip install -r requirements.txt
```

### How to Use VCD in LVLMs

The two core function of VCD, adding noise to images and generating text based on VCD sampling, are found in the `vcd_utils` folder. Scripts for using VCD sampling in LLaVA, InstructBLIP, and QwenVL are located in `VCD/eval`. We have annotated some key changes with `## cd_comment` for easy location using ctrl+f.

To help you get started quickly, here's an example using LLaVA on how to replace the conventional sampling method with the VCD method during generation:
1. Add the following at the beginning of the start-up script:
```python
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()
```
The `evolve_vcd_sampling` function replaces the sampling function in the transformers library. The modified sampling function includes an option for visual contrastive decoding, while keeping the rest unchanged.

2. Slightly modify `llava_llama.py`:

   a. Add contrastive decoding parameters in the `LlavaLlamaForCausalLM` class's `forward` function to avoid exceptions in `model.generate`.
   
   b. Add the `prepare_inputs_for_generation_cd` function.

3. Add noise to the image:
```python
from vcd_utils.vcd_add_noise import add_diffusion_noise
image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
```
set the hyperparameter in the `generate` function:
```python
output_ids = model.generate(
    input_ids,
    images=image_tensor.unsqueeze(0).half().cuda(),
    images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
    cd_alpha = args.cd_alpha,
    cd_beta = args.cd_beta,
    do_sample=True)
```

## 🏅 Experiments
- **MHR significantly mitigates the multilingual hallucination issue across different languages.**
![exp1](figs/pope_res.jpg)
*table 1. Enhanced LLaVA 1.5 model Performances on POPE benchmark’s all 3 datasets. We select the “popular" type to test.
Average scores of current partition are marked in <mark style="background-color: gray"> gray </mark> and bold text denotes the best results of the same backbone*

- **MHR gain remarkable performance on MME hallucination subset**
![exp2](figs/mme_res.jpg)
*table 2. Results on the hallucination subset of MME. Higher scores indicate better performance and fewer hallucinations. The
best performances within each setting are bolded. Limited by space, we only present 4 languages here, including high-resource
languages ru and zh, and low-resource languages uk and bg. To help understand the overall performance comparison, we also
report the average results for all 13 languages.*

![exp2](figs/mme_res.jpg)
*figure 2. The performance on the full MME set, which consists of 14 tasks. Each graph displays the performance of the
respective LLaVA-1.5 and our MHR model. Here we present results in four languages (uk, zh, bg, and ru) as outlined in Table 2.*

- **Please refer to our paper for detailed experimental results.**



## 📌 Examples
![Case1](figs/qualitive.jpg)
*figure 5. Illustration of hallucination removal by our proposed MHR with 7 languages as an example. We mark the hallucination part of response by <mark style="background-color: yellow"> Yellow </mark> and correctness by <mark style="background-color: green"> Green </mark>*




## 📝 Related Projects
- [Contrastive Decoding](https://github.com/XiangLi1999/ContrastiveDecoding): Open-ended Text Generation as Optimization
- [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip): Towards General-purpose Vision-Language Models with Instruction Tuning
- [LLaVA 1.5](https://github.com/haotian-liu/LLaVA): Improved Baselines with Visual Instruction Tuning
- [VCD](https://github.com/DAMO-NLP-SG/VCD):VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding
- [HA-DPO](https://github.com/opendatalab/HA-DPO):HA-DPO (Hallucination-aware Direct Preference Optimization) 
