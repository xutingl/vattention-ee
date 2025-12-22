# Introduction

vAttention is a memory manager for KV-cache in LLM serving systems. It decouples the allocation of virtual memory and physical memory using the [CUDA virtual memory APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html). This approach enables allocating physical memory on demand while retaining the contiguity of KV-cache in virtual memory. This way, vAttention provides support for dynamic memory allocation to unmodified attention kernels. This way of memory management is different from the popular [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) approach; PagedAttention implements demand paging in user space and requires rewriting custom kernels to support dynamic memory allocation. vAttention also improves performance over PagedAttention in many cases, especially for prefill-bound workloads. Please checkout our [paper](https://arxiv.org/abs/2405.04437) for more details.


# Content

This repository contains an implementation of vAttention, intergrated with an LLM serving system Sarathi-Serve that was published in OSDI'24 ([paper](https://www.usenix.org/conference/osdi24/presentation/agrawal), [code](https://github.com/microsoft/sarathi-serve)). The content is organized as follows:

 * `vattention` contains the source code of vattention memory allocator
 * `sarathi-lean` modified version of Sarathi-Serve with support for both PagedAttention and vAttention style of memory management
 * `scripts` contains scripts to run the experiments
 * `nvidia-vattn-uvm-driver` contains our modified version of the NVIDIA UVM drivers
 * `microbenchmarks` contains scripts to run some useful microbenchmarks


# Installation and Dependencies

Using this repo requires **PyTorch 2.3.0** and **CUDA 12.1** (or later but other CUDA versions may or may not work). We have tested vAttention with the Linux kernel, **A100 GPUs** and **python 3.10** but expect it to work on other Linux-based systems as long as they are running the specified CUDA and PyTorch versions.

To install vAttention and Sarathi-Serve, create a conda environment as follows:

```sh
conda create -n vattn python=3.10
conda activate vattn
```

Now, download and extract libtorch first (this is required to build the vattention memory allocator), and then build sarathi-serve and vattention as follows:

```sh
# the libtorch version has to match with the torch version, and we have tested only v2.3.0
wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.3.0%2Bcu121.zip
unzip libtorch-shared-with-deps-2.3.0+cu121.zip

# build sarathi-serve
cd sarathi-lean/
pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
cd ../

# build vattention
cd vattention/
LIBTORCH_PATH=<path to libtorch dir> python setup.py install
cd ../
```
# Running Benchmarks

The repo provides a benchmark-runner which can be used to run different workloads (dynamic/static, datasets/synthetic) with various attention-backends and schedulers. The benchmark-runner provides a comprehensive list of configuration knobs listed in [default.yml](sarathi-lean/sarathi/benchmark/config/default.yml). Please check [Sarathi-Serve](sarathi-lean/sarathi/benchmark/README.md) for a detailed explanation of the knobs.

This repository includes two customizable benchmark scripts:

* [benchmark_e2e_dynamic_trace.py](scripts/benchmark_e2e_dynamic_trace.py): This script runs expriments on a dynamic trace. It runs 256 requests from the **_arxive dataset_** for qps of 0.4, 0.8, 1, 2, 4 and 6 where requests arrive as per the poisson distribution.
* [benchmark_e2e_static_trace.py](scripts/benchmark_e2e_static_trace.py): This script runs experiments on a static trace and can be used to reproduce the makespan results of our paper. It runs 50 requests for context length 32k, 64k and 128k and prefill to decode ratios of 500, 100 and 50.

```sh
# testing the setup:
python scripts/benchmark_e2e_static_trace.py --test
or 
python scripts/benchmark_e2e_dynamic_trace.py --test

# run benchmarks for performance evaluation:
python scripts/benchmark_e2e_static_trace.py
or
python scripts/benchmark_e2e_dynamic_trace.py
```

Benchmark results are redirected to `experiments/e2e_static_eval` or `experiments/e2e_dynamic_eval`. Model configurations can be found in `scripts/utils.py` (all Yi and Llama family of models are expected to work). Parse benchmark results as follows:

```sh
python scripts/process_e2e_static.py
or
python scripts/process_e2e_dynamic.py
```


### Configuring attention backends

We have modified Sarathi-Serve to support [FlashAttention]((https://github.com/Dao-AILab/flash-attention)) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer) backends for attention computation.

* [vattention_flashattention_wrapper.py](sarathi-lean/sarathi/model_executor/attention/vattention_flashattention_wrapper.py) This backend uses FlashAttention's `flash_attn_with_kvcache` API for both prefill and decode attention computation.
* [vattention_flashinfer_wrapper.py](sarathi-lean/sarathi/model_executor/attention/vattention_flashinfer_wrapper.py) This experimental backend demonstrates the portability of vAttention approach. It uses FlashInfer's `flashinfer.prefill.single_prefill_with_kv_cache` API for (non-paged) prefill and FlashAttention's `flash_attn_with_kvcache` API for (non-paged) decode.


The backends can be configured by updating the `attention_backends` list in our scripts. We currently support `fa_paged_[block_size]`, `fi_paged_[block_size]`, `fa_vattn_[page_size]`, `fi_vattn_[page_size]`, `fa_vattn_[page_size]_sync`, `fi_vattn_[page_size]_sync` where `fa` denotes FlashAttention (we tested v2.5.9) and `fi` denotes FlashInfer (we tested v0.0.6). We recommend using block size 256 for FlashAttention and 16 for FlashInfer because we have observed them performing best with these block sizes. vAttention supports 64KB, 128KB, 256KB and 2MB page sizes (example knobs: `fa_vattn_256kb`, `fi_vattn_2mb_sync`). Using suffix `_sync` in vAttention knob disables our optimization of overlapping memory allocation with compute which may be useful for benchmarking. We recommend using vAttention with asynchronous memory allocation i.e., without the `_sync` suffix.


### Using smaller page sizes

NVIDIA CUDA drivers allocate memory only at the granularity of large pages (2MB or above). If you want to use vAttention with smaller page sizes of 64KB, 128KB or 256KB, please follow the [README.md](./nvidia-vattn-uvm-driver/README.md) to replace the default CUDA UVM driver with our custom driver (check `nvidia-vattn-uvm-driver`).

**NOTE:** Replacing CUDA drivers is not required if you want to use vAttention with only 2MB pages.

# OpenAI Compatible API

We also provide an OpenAI compatible API to facilitate benchmarking. An endpoint such as this one can be used with LLM benchmarking tools like [metron](https://github.com/project-metron/metron/tree/main?tab=readme-ov-file).
Start the server as follows:

```sh
cd sarathi-lean/
python -m sarathi.entrypoints.openai_server.api_server [COMMAND LINE ARGUMENTS]

# for example, run Yi-6B on a single GPU with fa_paged attention backend
python -m sarathi.entrypoints.openai_server.api_server --model_name 01-ai/Yi-6B-200k --model_tensor_parallel_degree 1 --model_attention_backend fa_paged --model_block_size 256
# or, run Llama-3-8B on two GPUs with fa_vattn attention backend using 2MB pages
python -m sarathi.entrypoints.openai_server.api_server --model_name meta-llama/Meta-Llama-3-8B --model_tensor_parallel_degree 2  --model_attention_backend fa_vattn --model_block_size 2097152
```
Just like the benchmark runner, you can configure many other knobs listed here: [default.yml](sarathi-lean/sarathi/benchmark/config/default.yml). Once the serve is up and running, you can use metron to benchmark performance as follows:

```sh
# Export API Key and URL
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE=http://localhost:8000/v1

# running a static trace
 python -m metron.run_benchmark \
--model "01-ai/Yi-6B-200k" \
--max-num-completed-requests 150 \
--timeout 600 \
--num-ray-clients 2 \
--num-concurrent-requests-per-client 5 \
--output-dir "experiments" \
--request-interval-generator-provider "static" \
--request-length-generator-provider "fixed" \
--fixed-request-generator-prefill-tokens 65536 \ 
--fixed-request-generator-decode-tokens 128 \
--request-generator-max-tokens 65536 \

# running a dynamic trace
python -m metron.run_benchmark \
--model "01-ai/Yi-6B-200k" \
--max-num-completed-requests 150 \
--timeout 600 \
--num-ray-clients 2 \
--num-concurrent-requests-per-client 5 \
--output-dir "experiments" \
--request-interval-generator-provider "poisson" \
--poisson-request-interval-generator-qps 0.5 \
--request-length-generator-provider "trace" \
--trace-request-length-generator-trace-file "sarathi-lean/data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv" \
--request-generator-max-tokens 8192 \
```

The results would be redirected to `experiments` directory. You can learn more about customising the benchmarks you run here: [project metron](https://project-metron.readthedocs.io/en/latest/).


# Using vAttention APIs for memory management in LLM serving

vAttention exports a set of simple APIs that a serving system can use for KV-cache related memory management. We choose Sarathi-Serve to exemplify this because Sarathi-Serve is a state-of-the-art LLM inference scheduler, has an elaborate metric store and a versatile benchmark_runner that makes running traces and performing experiments easy. Furthermore, its modular setup makes it easy to add more attention backends. Our core APIs are used as follows in Sarathi-Serve:

- [vATTN_cache_engine.py](sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py): The `vATTNCacheEngine` class initializes and manages some aspects of KV-cache in python land e.g., mapping the sequence id of a request to its batch index in the KV-cache, and the current context length of the each request (like `vLLMCacheEngine`). vAttention memory allocator is initialized as follows:

    ```sh
    vattention.init_kvcache(
        self.num_layers,
        self.num_heads,
        self.head_size,
        self.max_batch_size,
        self.max_model_seq_len,
        self.device_idx,
        self.dtype,
        self.page_size
        )
    ```

    which returns **virtual** PyTorch tensors without any physical memory mapped underneath. The serving system can also reserve physical memory for KV-cache ahead-of-time as follows:

    ```sh
    vattention.reserve_physical_pages(cache_config.memory_for_gpu)
    ```

    which pre-allocates physical memory pages on the GPU. These pages are then attached to the virtual tensors at runtime.
    
    When a request is scheduled for the first time, `vATTNCacheEngine` calls the vattention memory allocator's `alloc_new_batch_idx` to get its batch index which determines the request's KV-cache offset within the virtual tensors.

- [base_worker.py](sarathi-lean/sarathi/worker/base_worker.py): Before the model forward pass, the worker calls

    ```sh
        # asynchronous memory allocation
        vattention.step_async(self.curr_seq_lens)
        or
        # synchronous memory allocation
        vattention.step(self.curr_seq_lens)
    ```

    which allocates physical memory pages for the active requests, based on their requirement. After the forward pass, the worker calls

    ```sh
        vattention.free_batch_idx(batch_idx)
    ```

    on request ids which have been completed. vAttention can reclaim these pages as and when required.


And that is most of it.

# Installation Guide for RTX 5060 Ti / Blackwell GPUs (sm_120)

This guide documents the working installation steps for RTX 5060 Ti and other Blackwell architecture GPUs (compute capability 12.0).

## Prerequisites

- GPU: RTX 5060 Ti or other Blackwell GPU (sm_120)
- CUDA Toolkit: 12.8+ (nvcc 12.9+ recommended)
- Python: 3.10
- OS: Linux

## Step-by-Step Installation

### 1. Create Conda Environment

```sh
conda create -n vattn python=3.10
conda activate vattn
```

### 2. Install PyTorch 2.7.0 with CUDA 12.8

```sh
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify installation:
```sh
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Compute capability: {torch.cuda.get_device_capability(0)}')"
```

Expected output should show `Compute capability: (12, 0)`.

### 3. Download libtorch (for vattention)

```sh
wget https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.7.0%2Bcu128.zip
unzip libtorch-shared-with-deps-2.7.0+cu128.zip -d libtorch_270
export LIBTORCH_PATH=$(pwd)/libtorch_270/libtorch
```

### 4. Install flash-attn 2.8.3

flash-attn 2.8.3 supports sm_120. Earlier versions (e.g., 2.5.9) do not.

```sh
pip uninstall flash-attn -y
export TORCH_CUDA_ARCH_LIST="12.0"
pip install flash-attn==2.8.3 --no-build-isolation --no-deps
```

Verify:
```sh
python -c "from flash_attn import flash_attn_with_kvcache; print('flash-attn installed successfully')"
```

### 5. Install flashinfer

```sh
pip install flashinfer-python --no-deps --no-build-isolation --index-url https://flashinfer.ai/whl/cu128/torch2.7/
```

### 6. Build sarathi with sm_120 support

**Important:** sarathi's `setup.py` needs modification to support sm_120:

1. Edit `sarathi-lean/setup.py`:
   - Line 44: Change `valid_caps = {70, 75, 80, 86, 89, 90}` to `valid_caps = {70, 75, 80, 86, 89, 90, 120}`
   - After line 73, add:
     ```python
     if 120 in compute_capabilities and nvcc_cuda_version < Version("12.8"):
         raise RuntimeError(
             f"CUDA 12.8 or higher is required for GPUs with compute capability 12.0 (Blackwell). "
             f"Found CUDA {nvcc_cuda_version}."
         )
     ```
   - Replace the NVCC version detection section (around line 54) to use system nvcc:
     ```python
     # Validate the NVCC CUDA version.
     # Try system nvcc first (might be newer than CUDA_HOME)
     nvcc_cuda_version = None
     import shutil
     system_nvcc = shutil.which("nvcc")
     if system_nvcc:
         try:
             nvcc_output = subprocess.check_output([system_nvcc, "-V"], universal_newlines=True)
             output = nvcc_output.split()
             release_idx = output.index("release") + 1
             nvcc_cuda_version = parse(output[release_idx].split(",")[0])
             print(f"Using system nvcc version {nvcc_cuda_version}")
         except:
             pass
     
     # Fallback to CUDA_HOME nvcc
     if nvcc_cuda_version is None:
         nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
     ```

2. Build sarathi:
```sh
cd sarathi-lean/
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))  # Use conda's CUDA toolkit
rm -f sarathi/*.so  # Remove old compiled extensions
pip install -e . --no-build-isolation
cd ../
```

Verify sarathi was compiled for sm_120:
```sh
python -c "import sarathi; print('sarathi installed successfully')"
```

### 7. Build vattention (optional, if using vattention backends)

If you plan to use `fa_vattn_*` or `fi_vattn_*` attention backends:

1. Edit `vattention/setup.py`:
   - Update `LIBTORCH_PATH` to point to your libtorch directory (or use environment variable)

2. Build vattention:
```sh
cd vattention/
export TORCH_CUDA_ARCH_LIST="12.0"  # or "9.0" if your CUDA toolkit < 12.8
python setup.py install
cd ../
```

**Note:** If your system CUDA toolkit is < 12.8, you may need to compile for sm_90 as a fallback:
```sh
export TORCH_CUDA_ARCH_LIST="9.0"
```

## Configuration Adjustments

### GPU Memory Settings

For RTX 5060 Ti (16GB), adjust these settings in `scripts/run_ee.py`:

```python
gpu_mem_util = 0.85  # Lowered from 0.99 to leave room for KV cache
max_tokens = 512     # Lowered from 1024 to fit in available GPU blocks
```

### Running the Benchmark

```sh
cd ~/vattention-ee
HF_HUB_OFFLINE=1 CUDA_HOME=$(dirname $(dirname $(which nvcc))) \
python scripts/run_ee.py \
  --ee_policy=rebatching \
  --max_batch_size=1 \
  --num_requests=20 \
  --shallow_exit_layer=32 \
  --conf_threshold=0.6
```

## Troubleshooting

### "FlashAttention only supports Ampere GPUs or newer"
- Ensure flash-attn 2.8.3 is installed (not 2.5.9)
- Restart Ray workers to pick up new flash-attn installation

### "no kernel image is available for execution on the device"
- Verify sarathi was rebuilt: `ls -lh sarathi-lean/sarathi/*.so` should show recent timestamps
- Check that `setup.py` includes `120` in `valid_caps`
- Ensure `CUDA_HOME` points to conda's CUDA toolkit (with nvcc 12.9+)

### "Not enough available memory"
- Lower `gpu_memory_utilization` to 0.75-0.85
- Reduce `max_model_len` to 512 or lower
- Use `max_batch_size=1`

### "CUDA 12.8 or higher is required"
- Check nvcc version: `nvcc --version` (should be 12.8+)
- Set `CUDA_HOME` to conda's CUDA: `export CUDA_HOME=$(dirname $(dirname $(which nvcc)))`

## Verified Working Versions

- PyTorch: 2.7.0+cu128
- flash-attn: 2.8.3
- flashinfer-python: 0.5.3
- sarathi: 0.1.7 (editable install from sarathi-lean/)
- CUDA Toolkit: 12.9 (via conda)
- Python: 3.10

## Summary

The key differences from the standard installation:
1. **PyTorch 2.7.0+cu128** (instead of 2.3.0+cu121) for sm_120 support
2. **flash-attn 2.8.3** (instead of 2.5.9) for sm_120 support
3. **Modified sarathi setup.py** to include sm_120 in valid compute capabilities
4. **Lower GPU memory utilization** (0.85 instead of 0.99)
5. **Reduced max_model_len** (512 instead of 1024) for 16GB GPUs


## Citation

If you use our work, please consider citing our paper:

```
@misc{prabhu2024vattention,
      title={vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention},
      author={Ramya Prabhu and Ajay Nayak and Jayashree Mohan and Ramachandran Ramjee and Ashish Panwar},
      year={2024},
      url={https://arxiv.org/abs/2405.04437},
}
```

## Acknowledgment

This repository originally started as a fork of [Sarathi-Serve](https://github.com/microsoft/sarathi-serve) which in turn is a fork of the [vLLM project](https://vllm-project.github.io/). vAttention and Sarathi-Serve are research prototypes and do not have complete feature parity with open-source vLLM. We have only retained the most critical features and adopted the codebase for faster research iterations.

## Vattention-EE 

### Run
```shell
python scripts/run_ee.py --ee_policy=rebatching --max_batch_size=4  --num_requests=20 --shallow_exit_layer=32 --conf_threshold=0.6 > outputs_13b/req_20_batch_4/rebatching.txt
```


