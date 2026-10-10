# Comprehensive Survey on Efficient Inference and Small Language Models

## TL;DR
- Autoregressive LLM generation splits execution into a prefill phase (dominated by large GEMMs) and a decoding phase (dominated by small GEMVs) [1].
- Edge deployments are constrained by weight-storage memory limits and high energy costs for loading weights, inspiring innovative broadcasting techniques [2].
- Advanced low-bit and intermediate quantization frameworks—such as W4A8KV4, BPDQ, and FlexQ—mitigate dequantization overhead and preserve accuracy [3][4][5].
- Small Language Models (SLMs) emerge as a vital new paradigm, delivering efficient alternatives to large-scale transformers in resource-constrained environments [6].

## Background
The rapid proliferation of Large Language Models (LLMs) has established them as foundational components of modern artificial intelligence applications [1]. However, deploying these massive architectures introduces substantial challenges regarding inference efficiency, memory footprint, and computational latency [1]. As models scale to support long-context windows, traditional GPU serving stacks are pushed into an attention-bound regime where both compute and memory overheads are dominated by the softmax-GEMM pipeline [7]. Addressing these computational and resource bottlenecks requires holistic innovation spanning algorithmic quantization, hardware-software co-design, and the deployment of purpose-built compact architectures [8][6].

## Serving Architecture and Phase Dynamics
Autoregressive generation divides model execution into two distinct operational phases, each presenting unique performance bottlenecks [1]. The prefill phase is predominantly characterized by large General Matrix-Matrix multiplications (GEMMs), whereas the decoding phase is driven by small General Matrix-Vector multiplications (GEMVs) [1]. Modern serving infrastructures require shape-adaptive architectures and disaggregated optimization strategies to handle these divergent computational profiles effectively [1]. Furthermore, long-context inference exacerbates memory bandwidth pressures, driving the need for specialized kernels that optimize the execution pipeline on advanced hardware accelerators [7].

## Hardware Acceleration and Attention Optimization
Hardware-efficient optimizations are critical for sustaining high throughput under extreme inference demands [7]. On NVIDIA Blackwell GPUs, for example, FP4 Tensor Cores provide exceptionally high matrix multiplication throughput [7]. Nevertheless, fully utilizing FP4 precision for attention mechanisms often fails without careful hardware-software integration [7]. To bridge this gap, train-free attention kernels—such as specialized QK4V8 FlashAttention mechanisms—are developed to navigate the attention-bound regime without imposing prohibitive training overheads [7].

## Advanced Quantization and System Co-Design
Quantization remains a primary vehicle for reducing memory footprints and accelerating serving throughput without severe degradation in model accuracy [3][4][5]. Several frameworks illustrate the evolution of this domain:
- **QServe:** Employs a W4A8KV4 quantization algorithm alongside system co-design principles to minimize dequantization overhead and boost serving throughput on GPUs [3].
- **BPDQ (Bit-Plane Decomposition Quantization):** Derives variable quantization grids from bit-planes and scalar coefficients, outperforming traditional low-bit quantization methods in resource-constrained contexts [4].
- **FlexQ:** Combines uniform weight quantization with adaptive activation retention and custom GPU kernels to optimize post-training INT6 inference efficiency and output fidelity [5].

## Edge Inference and Memory-Free Approaches
Expanding LLMs from centralized cloud environments to ubiquitous edge devices introduces severe physical constraints [2]. Typical edge hardware either lacks sufficient local memory to retain massive model weights or incurs prohibitive energy penalties simply loading weights into memory [2]. To circumvent these hardware limitations, novel memory-free edge inference approaches have been proposed, such as broadcasting AI weights over radio frequencies via RF computing [2].

## Small Language Models as Resource-Constrained Solutions
Complementing hardware and quantization innovations, Small Language Models (SLMs) represent a paradigm shift in natural language processing [6]. SLMs provide highly effective, lightweight alternatives to massive transformer models, specifically engineered for deployment in resource-constrained environments [6]. Scholarship examining SLMs highlights their viability across various domain-specific applications where latency, energy footprint, and memory constraints preclude the use of larger models [6].

## Trends and open problems
Current trends in efficient inference focus heavily on algorithm-system co-design, low-bit representations like W4A8KV4 and INT6 [3][5], variable-grid bit-plane decompositions [4], and hardware-tailored attention kernels for modern accelerators [7]. Prominent open problems include mitigating the energy and memory barriers of edge deployment [2], balancing the trade-offs between aggressive quantization and generative accuracy, and managing the distinct operational bottlenecks between prefill and decoding phases in long-context serving architectures [1].

## Source coverage

- Efficient learning and inference methods for large language models: This source contributes evidence to the survey. [8]

## References
[1] A Shape-Adaptive Architecture with Disaggregated Quantization for Efficient LLM Serving. arxiv. https://arxiv.org/abs/2610.07443 (2026-10-05)
[2] AIR-LLM: Broadcasting AI Weights over Radio for Memory-Free Edge LLM Inference via RF Computing. arxiv. https://arxiv.org/abs/2610.00465 (2026-09-30)
[3] QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving. hf-search. https://huggingface.co/papers/2405.04532 (2024-05-07)
[4] BPDQ: Bit-Plane Decomposition Quantization on a Variable Grid for Large Language Models. hf-search. https://huggingface.co/papers/2602.04163 (2026-02-04)
[5] FlexQ: Efficient Post-training INT6 Quantization for LLM Serving via Algorithm-System Co-Design. hf-search. https://huggingface.co/papers/2508.04405 (2025-08-06)
[6] Advances in Small Language Models: A Comprehensive Survey on Efficient NLP Solutions for Resource-Constrained Environments. web. https://doi.org/10.36227/techrxiv.175994221.11990581/v1 (2025-10-8)
[7] MpFA: Hardware-Efficient Train-Free QK4V8 FlashAttention Kernels on Blackwell GPUs. arxiv. https://arxiv.org/abs/2609.33135 (2026-09-27)
[8] Efficient learning and inference methods for large language models. web. https://doi.org/10.32657/10356/214513 (n.d.)
