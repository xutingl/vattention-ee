#!/bin/bash

echo "Starting sequential execution of early exit experiments..."

CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy eager --conf_threshold 0.9 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 40
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy eager --conf_threshold 0.9 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 60
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy average --conf_threshold 0.9 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 40
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy average --conf_threshold 0.9 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 60
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy rebatching --conf_threshold 0.9 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 40
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy rebatching --conf_threshold 0.9 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 60
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy off --conf_threshold 0.9 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 40
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy off --conf_threshold 0.9 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 60

CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy eager --conf_threshold 0.95 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 60
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy average --conf_threshold 0.95 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 40
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy average --conf_threshold 0.95 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 60
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy rebatching --conf_threshold 0.95 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 40
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy rebatching --conf_threshold 0.95 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 60
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy off --conf_threshold 0.95 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 40
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy off --conf_threshold 0.95 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 60

CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy eager --conf_threshold 0.8 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 40
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy eager --conf_threshold 0.8 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 60
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy average --conf_threshold 0.8 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 40
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy average --conf_threshold 0.8 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 60
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy rebatching --conf_threshold 0.8 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 40
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy rebatching --conf_threshold 0.8 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 60
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy off --conf_threshold 0.8 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 40
CUDA_VISIBLE_DEVICES=0 python run_ee.py --ee_policy off --conf_threshold 0.8 --num_requests 500 --max_batch_size 8 --shallow_exit_layer 60

echo "All experiments completed!"