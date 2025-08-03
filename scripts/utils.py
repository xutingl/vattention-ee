import os
import argparse

KB = 1024
MB = 1024 * KB

parser = argparse.ArgumentParser(description='Run e2e dynamic trace experiments')
parser.add_argument('--test', action='store_true', help='Run a test experiment')
parser.add_argument('--num_requests', '-n', type=int, default=1000, help='Number of requests')
parser.add_argument('--qps', '-q', type=float, default=0.5, help='QPS')
parser.add_argument('--max_batch_size', '-b', type=int, default=8, help='Max batch size')
parser.add_argument('--shallow_exit_layer_1', '-l1', type=int, default=16, help='Shallow exit layer 1')
parser.add_argument('--shallow_exit_layer_2', '-l2', type=int, default=32, help='Shallow exit layer 2')
parser.add_argument('--conf_threshold_1', '-c1', type=float, default=0.95, help='Confidence threshold 1')
parser.add_argument('--conf_threshold_2', '-c2', type=float, default=0.8, help='Confidence threshold 2')
parser.add_argument('--num_ee_threshold', type=int, default=-1, help='Minimum number of requests that must want to EE before partial early exit is triggered (for rebatching).')
parser.add_argument('--ee_policy', '-e', type=str, default='off', help='EE policy')
parser.add_argument('--early_exit_head_path', '-p', type=str, default="", help='Early exit head path')
parser.add_argument('--csv_path', type=str, default="/workspace/xutingl/vattention-ee/outputs_13b/req_100_batch_4_csv/", help='Path to save CSV results')
parser.add_argument('--kv_method', type=str, default='copy', help='Technique to fill missing kv cache')
parser.add_argument('--enable_profiling', action='store_true', help='Enable profiling')
parser.add_argument('--buffer_age_factor', '-a', type=float, default=0.0, help='Buffer age factor')
args = parser.parse_args()

src = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(src)
main = os.path.join(root, 'sarathi-lean', 'sarathi', 'benchmark', 'main.py')

static_experiment_dir = os.path.join(root, 'experiments', 'e2e_static_eval')
dynamic_experiment_dir = os.path.join(root, 'experiments', 'e2e_dynamic_eval')

# dataset_subpath = 'data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv'
# dataset_name = 'sharegpt'
dataset_subpath = 'data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv'
dataset_name = 'arxiv'

# upper limit on maximum context length for dynamic traces
MAX_CONTEXT_LENGTH_DYNAMIC_TRACES = 32768

models = {
    'yi-6b-1': {'tp': 1, 'hfrecord': '01-ai/Yi-6B-200k', 'logentry': 'yi-6b'},
    'llama-3-8b-1': {'tp': 1, 'hfrecord': 'meta-llama/Meta-Llama-3-8B' , 'logentry': 'llama-3-8b-ee-eager'},
    'llama-3-8b-2': {'tp': 2, 'hfrecord': 'meta-llama/Meta-Llama-3-8B' , 'logentry': 'llama-3-8b'},
    'yi-34b-2': {'tp': 2, 'hfrecord': '01-ai/Yi-34B-200k', 'logentry': 'yi-34b'},
    'llama-2-13b': {'tp': 1, 'hfrecord': 'meta-llama/Llama-2-13b-chat-hf', 'logentry': 'llama2-13b-ee'},
    # 'llama-2-70b': {'tp': 1, 'hfrecord': 'TheBloke/Llama-2-70B-GPTQ', 'logentry': 'llama2-70b-ee'}, # 4 bit quant
    'llama-2-70b': {'tp': 1, 'hfrecord': 'meta-llama/Llama-2-70b-chat-hf', 'logentry': 'llama2-70b-ee'},
    # 'llama-2-70b': {'tp': 1, 'hfrecord': 'caisarl76/llama2-70B-8bit', 'logentry': 'llama2-70b-ee'}, # 8 bit quant
    'llama-3-70b': {'tp': 1, 'hfrecord': 'meta-llama/Llama-3.3-70B-Instruct', 'logentry': 'llama3-70b-ee'},
    
    'llama-2-7b': {'tp': 1, 'hfrecord': 'meta-llama/Llama-2-7b-chat-hf', 'logentry': 'llama2-7b-ee'},
}

# vattention allocates memory in power of two while fa_paged/fi_paged
# sometimes run into illegal memory access if context length is more than
# a certain limit (e.g., 200K) and rope scaling is applied (observed for yi-6B).
# Hence, we set max context length as per the experiment to get around this problem.
# TODO(ashish) Fix it.
def get_max_context_length(attn_backend, context_len):
    if context_len & (context_len - 1) == 0:
        return context_len
    if '_paged' in attn_backend.lower():
        return min(200000, 2 ** (context_len.bit_length() + 1))
    return 2 ** (context_len.bit_length() + 1)

def get_paths():
    return src, root, main

def extract_substr(log, start_sub, end_sub):
    start_index = log.find(start_sub)
    if start_index == -1:
        return None
    start_index += len(start_sub)

    end_index = log.find(end_sub, start_index)
    if end_index == -1:
        return None

    return log[start_index:end_index].strip("/")

def get_output_files(experiment_dir, log_file='sequence_metrics.csv'):
    logs = []
    for dirpath, dirnames, filenames in os.walk(experiment_dir):
        for filename in filenames:
            log = os.path.join(dirpath, filename)
            if log.endswith(log_file):
                logs.append(log)
    return logs

def get_block_or_page_size(attn_backend):
    if '64kb' in attn_backend.lower():
        return 64 * KB
    elif '128kb' in attn_backend.lower():
        return 128 * KB
    elif '256kb' in attn_backend.lower():
        return 256 * KB
    elif '2mb' in attn_backend.lower():
        return 2 * MB
    elif 'fa_paged' in attn_backend.lower():
        return attn_backend.split('_')[-1]
    elif 'fi_paged' in attn_backend.lower():
        return attn_backend.split('_')[-1]
    else:
        raise ValueError(f"Unsupported attention backend: {attn_backend}")

def get_backend(attn_backend):
    if 'fa3_vattn' in attn_backend.lower():
        return 'fa3_vattn_sync' if '_sync' in attn_backend else 'fa3_vattn'
    if 'fa_vattn' in attn_backend.lower():
        return 'fa_vattn_sync' if '_sync' in attn_backend else 'fa_vattn_megacache'
    elif 'fi_vattn' in attn_backend.lower():
        return 'fi_vattn_sync' if '_sync' in attn_backend else 'fi_vattn'
    elif 'fa_paged' in attn_backend.lower():
        return 'fa_paged'
    elif 'fi_paged' in attn_backend.lower():
        return 'fi_paged'
    else:
        raise ValueError(f"Unsupported attention backend: {attn_backend}")
