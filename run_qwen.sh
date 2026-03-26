python scripts/run_ee.py --ee_policy=rebatching --max_batch_size=4  --num_requests=100 --csv_path="/workspace/xutingl/vattention-ee/outputs_qwen/14b" --shallow_exit_layer=30 --conf_threshold=0.7 --num_ee_threshold=1 > outputs_qwen/14b/req_100_batch_4_layer_30_conf_0.7_rebatching.txt

python scripts/run_ee.py --ee_policy=median --max_batch_size=4  --num_requests=100 --csv_path="/workspace/xutingl/vattention-ee/outputs_qwen/14b" --shallow_exit_layer=30 --conf_threshold=0.7 > outputs_qwen/14b/req_100_batch_4_layer_30_conf_0.7_median.txt

python scripts/run_ee.py --ee_policy=off --max_batch_size=4  --num_requests=100 --csv_path="/workspace/xutingl/vattention-ee/outputs_qwen/14b" --shallow_exit_layer=30 --conf_threshold=0.7 > outputs_qwen/14b/req_100_batch_4_off.txt

