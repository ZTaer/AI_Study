"""执行 Notebook 的三轮训练流水线，逐单元格保存结果，支持权重断点下载。

使用 course1_365 的 Python 执行。不要同时在浏览器重复启动训练。
源 Notebook 不被执行器覆盖；输出保存在 XLNet_TRAINING_RESULT.ipynb。
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
from pathlib import Path
import subprocess
import shutil
import hashlib
from datetime import datetime
import nbformat
from nbclient import NotebookClient
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'XLNet_TEST.ipynb'
RESULT = ROOT / 'XLNet_TRAINING_RESULT.ipynb'
REVISION = 'ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79'

def log(message):
    print(f'{datetime.now().isoformat(timespec="seconds")} {message}', flush=True)

def main():
    # 先取得官方 config 的缓存路径；完整权重就绪后才放入相同 snapshot。
    config = Path(hf_hub_download('xlnet/xlnet-base-cased', 'config.json', revision=REVISION))
    weights = config.parent / 'pytorch_model.bin'
    if not weights.exists():
        partial = ROOT / 'xlnet_weights.download'
        log('Downloading official pretrained weights (resume enabled). Training has not started.')
        # 慢连接主动重连，每次从文件当前大小续传，不丢弃已有数据。
        # 使用外层重试，确保每次 curl 都重新计算断点位置。
        for attempt in range(1, 21):
            log(f'Download attempt {attempt}/20; existing bytes: {partial.stat().st_size if partial.exists() else 0}')
            result = subprocess.run(['curl.exe', '-L', '--fail', '--continue-at', '-',
                '--connect-timeout', '20', '--speed-limit', '153600', '--speed-time', '30',
                '--max-time', '300', '--output', str(partial),
                f'https://huggingface.co/xlnet/xlnet-base-cased/resolve/{REVISION}/pytorch_model.bin'])
            if result.returncode == 0:
                break
            # 网络/超时错误可续传；HTTP、磁盘及不支持 Range 等错误立即暴露。
            if result.returncode not in {5, 6, 7, 18, 28, 35, 52, 55, 56}:
                result.check_returncode()
        else:
            raise RuntimeError('下载重试次数已用完，已保留断点文件；训练未启动。')
        # curl 成功才移动，不让不完整权重被模型加载器识别。
        # 下载目录和缓存可能分别在 D/C 盘，不能直接使用跨盘 os.replace。
        expected_hash = 'b13dc2d3664a385b92d087229d5410d9e3020ede976e7ff4c62c9c85cd969a42'
        with partial.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected_hash, '权重校验失败'
        staged = weights.with_suffix('.bin.staging')
        shutil.copyfile(partial, staged)
        with staged.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected_hash, '缓存复制校验失败'
        staged.replace(weights)  # 在同一缓存目录内原子替换。
    log('Weights available. Executing preparation, 3 epochs, evaluation, saving and inference.')
    notebook = nbformat.read(SOURCE, 4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(notebook, timeout=None, kernel_name='nlp_course_env',
        resources={'metadata': {'path': str(ROOT)}})
    try:
        with client.setup_kernel():
            for index, cell in enumerate(notebook.cells):
                log(f'Cell {index + 1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:100]}')
                client.execute_cell(cell, index)
                nbformat.write(notebook, RESULT)
    finally:
        nbformat.write(notebook, RESULT)
    log('COMPLETED: all 3 epochs, test metrics, saved model and inference. See result notebook.')

if __name__ == '__main__':
    main()
