"""在 /tmp 副本中演示三个错误怎样触发现有测试；不修改项目实现。"""
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MUTATIONS = [
    ('M1 页号除法误写成取模', 'block_manager.hpp',
     'const std::size_t logical_block = token_index / block_size_;',
     'const std::size_t logical_block = token_index % block_size_;'),
    ('M2 忽略本轮 Token 预算', 'scheduler.hpp',
     'const std::size_t count = std::min(sequence->pending_tokens(), budget);',
     'const std::size_t count = sequence->pending_tokens();'),
    ('M3 分配时多记一次引用', 'block_manager.hpp',
     'block.ref_count = 1;', 'block.ref_count = 2;'),
]


def run_no_core(binary, directory):
    def disable_core():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return subprocess.run([str(binary)], cwd=str(directory), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=20, preexec_fn=disable_core)


def main():
    with tempfile.TemporaryDirectory(prefix='zyf_learning_mutation_') as name:
        work = Path(name)
        (work/'dev').mkdir()
        (work/'mini_vllm').mkdir()
        original = {p.name: p.read_text() for p in (ROOT/'mini_vllm').glob('*.hpp')}
        shutil.copy2(ROOT/'dev/test_mini_vllm_control_plane.cpp', work/'dev/test.cpp')
        for case in [None] + MUTATIONS:
            for filename, contents in original.items():
                (work/'mini_vllm'/filename).write_text(contents)
            if case is not None:
                title, filename, before, after = case
                assert original[filename].count(before) == 1, '源码已变更，需要重新核对错误注入点'
                (work/'mini_vllm'/filename).write_text(original[filename].replace(before, after, 1))
            binary = work/'control_test'
            subprocess.run(['c++', '-std=c++17', '-O0', '-g', str(work/'dev/test.cpp'),
                            '-o', str(binary)], check=True, timeout=60,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            result = run_no_core(binary, work)
            if case is None:
                assert result.returncode == 0, result.stderr
                print('基线 PASS: 未修改的控制面测试正常通过')
            else:
                assert result.returncode != 0 and 'Assertion' in result.stderr, result.stderr
                diagnostic = result.stderr.strip().replace(str(work), '<临时目录>')
                print('%s: 被现有断言检测，returncode=%d' % (title, result.returncode))
                print(diagnostic)
    print('PASS: 三个错误各自编译成功、运行失败；临时副本已清理，项目源码未改动。')


if __name__ == '__main__':
    main()
