# Outlier-Safe Pre-Training

[![arXiv](https://img.shields.io/badge/arXiv-2506.19697-b31b1b?style=flat-square)](https://arxiv.org/abs/2506.19697)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97Hugging_Face-Collection-ffd200?style=flat-square)](https://huggingface.co/collections/dmis-lab/outlier-safe-pre-training-osp-685bda10aa1e8a19fcb58ea8)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github&style=flat-square)](https://github.com/dmis-lab/Outlier-Safe-Pre-Training)

This repository contains the evaluation code used in [Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](https://www.arxiv.org/abs/2506.19697). The codebase is heavily based on [QuaRot](https://github.com/spcl/QuaRot) and [SpinQuant](https://github.com/facebookresearch/SpinQuant). The code has been properly adapted for various quantization scenarios.

## Introduction

Quantization plays a crucial role in deploying Large Language Models (LLMs) in resource-constrained environments. However, the presence of outlier features significantly hinders low-bit quantization. While many studies address this problem in a post-hoc manner to make use of already pre-trained models, the importance of handling outliers during pre-training is often underestimated.

Our work, **Outlier-Safe Pre-Training (OSP)**, proposes a practical approach to training models that are robust to outliers from the start, without sacrificing performance or efficiency. Specifically, OSP focuses on the following goals:

1. 📈**Scaling to production-level training requirements**<br/>
Prior methods for quantization-friendly pre-training are often limited to small-scale experiments (e.g., models under 1B parameters or 100B tokens). In contrast, we train a 1.4B-parameter model on 1 trillion tokens, demonstrating that OSP is effective at production scale.

2. ⚡**Maintaining computational efficiency comparable to standard training**<br/>
A method that prevents outliers but significantly reduces efficiency is unlikely to gain adoption. OSP introduces only a ~2% slowdown while reducing GPU memory usage, making it appealing for those seeking to train quantization-friendly foundation models from scratch.

3. 🧩**Ensuring full compatibility with existing inference pipelines**<br/>
We prioritize compatibility with widely adopted inference frameworks such as vLLM and SGLang. Rather than introducing architectural changes that break compatibility, OSP preserves computational invariance, allowing models to be directly integrated into existing pipelines without additional effort.

<p align="center">
    <img src="./images/figure2.png" alt="drawing" width="700"/>
</p>

## News

- **2025-06-25**: Released **Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models** on [arXiv](https://www.arxiv.org/abs/2506.19697), with [GitHub](https://github.com/dmis-lab/Outlier-Safe-Pre-Training) and [models](https://huggingface.co/collections/dmis-lab/outlier-safe-pre-training-osp-685bda10aa1e8a19fcb58ea8).
- **2025-05-16**: Our paper has been accepted to ACL 2025! 🎉

## Model Checkpoints

### Final Models

The models were trained on 1 trillion tokens, following the pre-training recipe of [SmolLM](https://huggingface.co/blog/smollm). Specifically, training was conducted using the [smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), a mixture of FineWeb-Edu, Cosmopedia, and Python-Edu.

- [🤗 OSP-1.4B-1T-Adam](https://huggingface.co/dmis-lab/OSP-1.4B-1T-Adam): Trained on the standard Adam optimizer, without any modifications.
- [🤗 OSP-1.4B-1T-Muon-SSNorm-EmbProj](https://huggingface.co/dmis-lab/OSP-1.4B-1T-Muon-SSNorm-EmbProj): Trained on the OSP framework. This is our final model.


### Ablation Models

<table>
    <thead>
        <tr>
            <th rowspan="2">Model</th>
            <th rowspan="2">Optimizer</th>
            <th rowspan="2">SSNorm</th>
            <th rowspan="2">EmbProj</th>
            <th rowspan="2">Ex. Kurt.</th>
            <th rowspan="2">Had.</th>
            <!-- <th colspan="2">16-16-16</th> -->
            <th colspan="2">4-4-4</th>
        </tr>
        <tr>
            <!-- <th>Avg.</th>
            <th>PPL</th> -->
            <!-- <th>Avg.</th>
            <th>PPL</th>
            <th>Avg.</th>
            <th>PPL</th>
            <th>Avg.</th>
            <th>PPL</th> -->
            <th>Avg.</th>
            <th>PPL</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href="https://huggingface.co/dmis-lab/OSP-1.4B-100B-Adam">🤗 OSP-1.4B-100B-Adam</a></td>
            <td>Adam</td>
            <td>✗</td>
            <td>✗</td>
            <td>1818.56</td>
            <td>✗<br>✔</td>
            <!-- <td>41.5<br>41.5</td>
            <td>11.4<br>11.4</td> -->
            <!-- <td>39.7<br>40.2</td>
            <td>21.6<br>22.3</td>
            <td>39.7<br>40.3</td>
            <td>21.6<br>22.3</td>
            <td>26.5<br>27.2</td>
            <td>1e5<br>3e4</td> -->
            <td>26.8<br>26.9</td>
            <td>8e4<br>3e4</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/dmis-lab/OSP-1.4B-100B-Muon-Only">🤗 OSP-1.4B-100B-Muon-Only</a></td>
            <td>Muon&dagger;<br/>(w/o Adam)</td>
            <td>✗</td>
            <td>✗</td>
            <td>361.35</td>
            <td>✗<br>✔</td>
            <!-- <td>41.0<br>41.0</td>
            <td>11.7<br>11.7</td> -->
            <!-- <td>38.4<br>37.5</td>
            <td>14.8<br>15.4</td>
            <td>38.3<br>37.5</td>
            <td>14.8<br>15.4</td>
            <td>26.3<br>33.3</td>
            <td>1e6<br>24.5</td> -->
            <td>26.3<br>33.1</td>
            <td>8e5<br>24.8</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/dmis-lab/OSP-1.4B-100B-Muon">🤗 OSP-1.4B-100B-Muon</a></td>
            <td>Muon</td>
            <td>✗</td>
            <td>✗</td>
            <td>1575.12</td>
            <td>✗<br>✔</td>
            <!-- <td>41.5<br>41.5</td>
            <td>11.4<br>11.4</td> -->
            <!-- <td>40.0<br>40.6</td>
            <td>13.8<br>12.9</td>
            <td>40.0<br>40.6</td>
            <td>13.8<br>12.9</td>
            <td>29.4<br>38.6</td>
            <td>934.3<br>15.7</td> -->
            <td>29.0<br>38.4</td>
            <td>1e4<br>15.8</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/dmis-lab/OSP-1.4B-100B-Muon-SSNorm">🤗 OSP-1.4B-100B-Muon-SSNorm</a></td>
            <td>Muon</td>
            <td>✔</td>
            <td>✗</td>
            <td>66.69</td>
            <td>✗<br>✔</td>
            <!-- <td><strong>41.8</strong><br><strong>41.8</strong></td>
            <td><strong>11.2</strong><br><strong>11.2</strong></td> -->
            <!-- <td><strong>41.0</strong><br><strong>40.8</strong></td>
            <td>12.4<br>12.2</td>
            <td><strong>40.9</strong><br><strong>40.8</strong></td>
            <td>12.4<br>12.2</td>
            <td>36.6<br>38.6</td>
            <td>43.3<br>33.7</td> -->
            <td>36.4<br>38.3</td>
            <td>44.2<br>34.1</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/dmis-lab/OSP-1.4B-100B-Muon-EmbProj">🤗 OSP-1.4B-100B-Muon-EmbProj</a></td>
            <td>Muon</td>
            <td>✗</td>
            <td>✔</td>
            <td>703.23</td>
            <td>✗<br>✔</td>
            <!-- <td>40.0<br>40.0</td>
            <td>12.3<br>12.3</td> -->
            <!-- <td>38.4<br>39.2</td>
            <td>14.8<br>13.9</td>
            <td>38.4<br>39.3</td>
            <td>14.8<br>13.9</td>
            <td>31.0<br>36.3</td>
            <td>99.7<br>22.1</td> -->
            <td>30.4<br>36.2</td>
            <td>114.6<br>22.3</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/dmis-lab/OSP-1.4B-100B-Muon-SSNorm-EmbProj">🤗 OSP-1.4B-100B-Muon-SSNorm-EmbProj</a></td>
            <td>Muon</td>
            <td>✔</td>
            <td>✔</td>
            <td><strong>0.04</strong></td>
            <td>✗<br>✔</td>
            <!-- <td>41.4<br>41.4</td>
            <td><strong>11.2</strong><br><strong>11.2</strong></td> -->
            <!-- <td>40.6<br>40.5</td>
            <td><strong>12.2</strong><br><strong>12.1</strong></td>
            <td>40.6<br>40.5</td>
            <td><strong>12.2</strong><br><strong>12.1</strong></td>
            <td><strong>37.9</strong><br><strong>39.1</strong></td>
            <td><strong>19.4</strong><br><strong>13.4</strong></td> -->
            <td><strong>37.5</strong><br><strong>38.9</strong></td>
            <td><strong>19.6</strong><br><strong>13.5</strong></td>
        </tr>
    </tbody>
</table>
&dagger;Model configuration that disables decoupled embedding optimization by training with Muon optimizer without Adam optimization on embedding layers 


## Getting Started

### Environment Setup

To begin, install the libraries needed to evaluate the quantized model's performance. We recommend creating a conda environment using:
```bash
$ conda env create -f environment.yaml
```

### Evaluation on WikiText-2

We support evaluating WikiText-2 under various quantization settings:
1. **Round-to-nearest (RTN)**: This method reduces the bitwidth by rounding normalized values to their nearest baselines. To evaluate performance, run:
    ```bash
    $ bash scripts/1_eval_ppl_rtn.sh [model_path] 4 4 4
    ```
2. **FFN Hadamard Rotation**: Based on QuaRot, this method applies online Hadamard rotation within the feedforward (FFN) layers.
    ```bash
    $ bash scripts/2_eval_ppl_rtn_had.sh [model_path] 4 4 4
    ```
3. **GPTQ**: Uses GPTQ for weight quantization to achieve further improvements. FFN rotation is also applied.
    ```bash
    $ bash scripts/3_eval_ppl_gptq_had.sh [model_path] 4 4 4
    ```
4. **QuaRot**: A full replication of QuaRot, applying Hadamard rotation to all weight parameters while preserving computational equivalence.
    ```bash
    $ bash scripts/4_eval_ppl_quarot.sh [model_path] 4 4 4
    ```

The three integers (e.g. `4 4 4`) represent the number of bits used for weight, activation, and key-value quantization, respectively. Based on this setup, you can replicate the following table:

| Quantization | Adam | Muon (OSP) |
|:-------------|-----:|-----------:|
| RTN | 14475.51 | **45.92** |
| + FFN Had&ddagger; | 4794.00 | **19.27** |
| + GPTQ | 3723.46 | **14.29** |
| + QuaRot | 16.62 | **14.38** |
| + SpinQuant | 14.94 | **13.66** |

&ddagger;Only applies Hadamard transform to FFN hidden states.


### Evaluation on Lighteval Benchmarks

To further assess model performance on various tasks, including multiple-choice and open-ended questions, we integrate [lighteval](https://github.com/huggingface/lighteval) into our quantization framework. We evaluate on 10 benchmarks, as used in the paper:
- ARC
- CommonsenseQA
- GSM8K
- HellaSwag
- MMLU
- OpenBookQA
- PIQA
- SIQA
- TriviaQA
- WinoGrande

For the full implementation details and task definitions, refer to [lighteval_tasks.py](./lighteval_tasks.py) and [tasks.txt](./tasks.txt). You can customize the tasks by following [this tutorial](https://huggingface.co/docs/lighteval/adding-a-custom-task). To run the evaluation pipeline, use the following command:
```bash
$ python lighteval_ptq.py \
    [model_path] \
    --w_bits 4 \
    --a_bits 4 \
    --kv_bits 4 \
    --tasks tasks_tmp.txt
```

You can add the following arguments to enable additional configurations: FFN Hadamard Rotation (`--rotate_down_proj`), GPTQ (`--no_rtn`), and QuaRot (`--no_rtn --rotate`). The `--rotate` flag applies Hadamard rotation to all weight parameters, while `--rotate_down_proj` enables online rotation within the FFN layers only.

#### Using vLLM

For faster evaluation, we also provide a lighteval pipeline based on vLLM:
```bash
$ python lighteval_ptq_vllm.py \
    [model_path] \
    --w_bits 4 \
    --a_bits 4 \
    --kv_bits 4 \
    --tasks tasks_tmp.txt
```
**Note**: Key-value quantization is not supported.

###  Distributed Muon on TPUs

As described in the paper, we implement distributed Muon to accelerate Newton-Schulz orthogonalization using parallel computation. The full implementation can be found in [optimization.py](./optimization.py), and can be used as follows:
```python
from optimization import muon

tx = muon(
    learning_rate=5e-4,
    beta=0.95,
    steps=5,  # Number of Newton-Schulz iterations
    eps=1e-8,
    weight_decay=1e-2,
)
```

This is the SPMD (Single Program, Multiple Data) version of the Muon optimizer. The device mesh should follow the shape (`dp`, `op`, `fsdp`), representing data parallelism, optimizer parallelism, and fully sharded data parallelism, respectively. Note that `op` is orthogonal to fsdp, and we recommend setting op to evenly divide the number of transformer layers. This will be discussed in more detail later.

For example, on a TPU v4-512 Pod slice, the device mesh can be defined as:
```python
mesh = np.arange(jax.device_count()).reshape(2, 8, 16)
mesh = mesh_utils.create_device_mesh(mesh.shape, allow_split_physical_axes=True)
Mesh(mesh, axis_names=("dp", "op", "fsdp")).__enter__()
```

In this configuration:
- The model is replicated across 2 data-parallel groups (`dp`).
- Each group shards model parameters and optimizer states across 16 devices (`fsdp`).
- Orthogonalization is performed in parallel across 8 devices (`op`).

#### How it works?

The implementation has two key components:
1. Gradient grouping by shape for batched orthogonalization.
2. Sharding over the `op` axis to parallelize Newton-Schulz steps.

```python
# Gather gradients with the same shape so they can be orthogonalized together
# and then reassigned to their corresponding parameters.
new_updates, grad_groups, name_groups = {}, defaultdict(list), defaultdict(list)
for name, grad in flatten_dict(updates).items():
    if isinstance(grad, jax.Array):
        grad_groups[grad.shape].append(grad)
        name_groups[grad.shape].append(name)
    else:
        new_updates[name] = grad
```

Once the gradients are grouped by shape, they are stacked and strictly sharded across devices:

```python
# If gradients can be distributed across devices based on the optimizer
# parallelism rank, they will be stacked and orthogonalized in a single
# operation. Otherwise, they will be normalized individually.
for shape, grads in grad_groups.items():
    print(f"[*] Muon Parallelsim: {len(grads)} x {shape}")
    if len(shape) == 2 and (chunk_size := len(grads) // op_rank * op_rank) > 0:
        chunks, grads = jnp.stack(grads[:chunk_size]), grads[chunk_size:]
        chunks = jax.lax.with_sharding_constraint(chunks, P("op", "fsdp"))
        grads = list(batch_ortho_fn(chunks)) + list(map(batch_ortho_fn, grads))
    else:
        grads = list(map(batch_ortho_fn, grads))
    for name, grad in zip(name_groups[shape], grads):
        if len(grad.shape) > 0:
            grad = jax.lax.with_sharding_constraint(grad, P("fsdp"))
        new_updates[name] = grad
```

If the stacked shape is not divisible by `op`, the mismatched gradients are orthogonalized by iterating manually. This design enables Muon to leverage optimizer parallelism effectively, achieving up to **97.9%** of the speed of standard Adam.

## Citation

```bibtex
@article{park2025osp,
      title={Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models}, 
      author={Jungwoo Park and Taewhoo Lee and Chanwoong Yoon and Hyeon Hwang and Jaewoo Kang},
      year={2025},
      eprint={2506.19697},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.19697}, 
}
```
