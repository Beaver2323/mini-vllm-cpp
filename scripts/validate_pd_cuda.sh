#!/usr/bin/env bash
# 从 llm.c 仓库根目录运行；先 conda activate zyf1。需要两张 CUDA GPU 和 GPT-2 checkpoint。
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
result_dir="${1:-benchmark/results/pd_reproduction}"
mkdir -p "$result_dir"
make test_minivllm_control_plane test_gpt2_cuda_sample_rows test_gpt2_cuda_model_runner \
  test_gpt2_cuda_prefix_cache test_gpt2_pd_engine benchmark_gpt2_cuda_prefix_cache \
  benchmark_gpt2_pd_serving benchmark_gpt2_cuda_serving GPU_COMPUTE_CAPABILITY="${GPU_COMPUTE_CAPABILITY:-86}"
# 环境与源码快照在计时外记录；即使工作区尚未提交也能定位实际实现。
python3 - "$result_dir" <<'PY'
import datetime, hashlib, json, os, pathlib, subprocess, sys
files = sorted(pathlib.Path('mini_vllm').rglob('*.hpp')) + sorted(pathlib.Path('mini_vllm/cuda').glob('*.cu*'))
files += [pathlib.Path('train_gpt2.cpp')] + sorted(pathlib.Path('benchmark').glob('*gpt2*cu'))
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
metadata = {
    '说明': '源码和 checkpoint 哈希标识实际运行版本；git_head 可能是未提交改动的父提交。',
    'utc_time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'git_head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
    'git_status': subprocess.check_output(['git','status','--short'], text=True),
    'omp_threads': os.environ['OMP_NUM_THREADS'],
    'checkpoint_sha256': digest(pathlib.Path('gpt2_124M.bin')),
    'source_sha256': {str(p): digest(p) for p in files},
    'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version','--format=csv'], text=True),
    'topology': subprocess.check_output(['nvidia-smi','topo','-m'], text=True),
    'nvcc': subprocess.check_output(['nvcc','--version'], text=True),
}
(pathlib.Path(sys.argv[1])/'environment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
PY
./test_minivllm_control_plane
./test_gpt2_cuda_sample_rows
./test_gpt2_cuda_sample_rows --fp32
./test_gpt2_cuda_model_runner --precision fp32 --disable-fusion
./test_gpt2_cuda_model_runner --precision fp16 --cuda-graph
./test_gpt2_cuda_model_runner --precision fp16 --full-logits --disable-fusion
./test_gpt2_cuda_prefix_cache
./test_gpt2_pd_engine
./test_gpt2_pd_engine --cuda-graph
./benchmark_gpt2_cuda_prefix_cache "$result_dir/prefix.csv" 7
./benchmark_gpt2_pd_serving "$result_dir/pd.csv"
./benchmark_gpt2_pd_serving "$result_dir/pd_graph.csv" --cuda-graph
for variant in full pruned full_graph pruned_graph; do
  options=()
  case "$variant" in full*) options+=(--full-logits);; esac
  case "$variant" in *graph) options+=(--cuda-graph);; esac
  ./benchmark_gpt2_cuda_serving --precision fp16 --repeats 7 "${options[@]}" \
    --json "$result_dir/rows_${variant}.json" --csv "$result_dir/rows_${variant}.csv"
done
compute-sanitizer --tool memcheck --error-exitcode 99 ./test_gpt2_pd_engine --cuda-graph
compute-sanitizer --tool memcheck --error-exitcode 99 ./test_gpt2_cuda_sample_rows
