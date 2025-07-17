# python run_ee.py --ee_policy=rebatching --max_batch_size=4  --num_requests=100 --shallow_exit_layer=60 --conf_threshold=0.9 > ../outputs_70b_llama3/req_100_batch_4_conf_09/rebatching.txt
# python run_ee.py --ee_policy=eager --max_batch_size=4  --num_requests=100 --shallow_exit_layer=60 --conf_threshold=0.9 > ../outputs_70b_llama3/req_100_batch_4_conf_09/eager.txt
# python run_ee.py --ee_policy=average --max_batch_size=4  --num_requests=100 --shallow_exit_layer=60 --conf_threshold=0.9 > ../outputs_70b_llama3/req_100_batch_4_conf_09/average.txt
# python run_ee.py --ee_policy=lazy --max_batch_size=4  --num_requests=100 --shallow_exit_layer=60 --conf_threshold=0.9 > ../outputs_70b_llama3/req_100_batch_4_conf_09/lazy.txt
# python run_ee.py --ee_policy=off --max_batch_size=4  --num_requests=100 --shallow_exit_layer=60 --conf_threshold=0.9 > ../outputs_70b_llama3/req_100_batch_4_conf_09/off.txt
# python run_ee.py --ee_policy=eager --max_batch_size=1  --num_requests=100 --shallow_exit_layer=60 --conf_threshold=0.9 > ../outputs_70b_llama3/req_100_batch_4_conf_09/ee_batch1.txt



# python scripts/run_ee.py --ee_policy=eager --max_batch_size=1  --num_requests=100 --shallow_exit_layer=24 --conf_threshold=0.9 --early_exit_head_path "/workspace/xutingl/finetune-ee/output/early_exit_head.pt"  > outputs_13b_tuned/req_100_batch_4_conf09/ee_nobatch_tunehead_nocopy.txt

python scripts/run_ee.py --ee_policy=rebatching --max_batch_size=4  --num_requests=100 --shallow_exit_layer=25 --conf_threshold=0.8 --csv_path="/workspace/xutingl/vattention-ee/outputs_13b/" --kv_method="copy" > outputs_13b/req_100_batch_4_conf_08_layer_25_rebatching_copy.txt


nsys profile -w true -t cuda,osrt -s cpu --cudabacktrace=true -x true -o nsight_profiles/rebatching_copy_nsight --python-sampling=true --python-sampling-frequency=2 python scripts/run_ee.py --ee_policy=rebatching --max_batch_size=2  --num_requests=4 --shallow_exit_layer=24 --conf_threshold=0.6 --csv_path="/workspace/xutingl/vattention-ee/outputs_13b/" --kv_method="copy" > outputs_13b/req_4_batch_2_conf_06_layer_24_rebatching_copy.txt

nsys profile -w true -t cuda,osrt -s cpu --cudabacktrace=true -x true -o nsight_profiles/rebatching_nocopy_nsight --python-sampling=true --python-sampling-frequency=2 python scripts/run_ee.py --ee_policy=rebatching --max_batch_size=2  --num_requests=4 --shallow_exit_layer=24 --conf_threshold=0.6 --csv_path="/workspace/xutingl/vattention-ee/outputs_13b/" --kv_method="copy" > outputs_13b/req_4_batch_2_conf_06_layer_24_rebatching_nocopy.txt

nsys profile -w true -t cuda,osrt -s cpu --cudabacktrace=true -x true -o nsight_profiles/average_nocopy_nsight --python-sampling=true --python-sampling-frequency=2 python scripts/run_ee.py --ee_policy=average --max_batch_size=2  --num_requests=4 --shallow_exit_layer=24 --conf_threshold=0.6 --csv_path="/workspace/xutingl/vattention-ee/outputs_13b/" --kv_method="copy" > outputs_13b/req_4_batch_2_conf_06_layer_24_average_nocopy.txt

nsys profile -w true -t cuda,osrt -s cpu --cudabacktrace=true -x true -o nsight_profiles/off_nsight --python-sampling=true --python-sampling-frequency=2 python scripts/run_ee.py --ee_policy=off --max_batch_size=2  --num_requests=4 --shallow_exit_layer=24 --conf_threshold=0.6 --csv_path="/workspace/xutingl/vattention-ee/outputs_13b/" --kv_method="copy" > outputs_13b/req_4_batch_2_off.txt