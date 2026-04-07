# quick_test.py - 低显存友好的快速验证（默认仅CPU）
import sys
sys.argv = [
    'main.py',
    '--algo', '5',
    '--batch_size', '1',
    '--num_epochs', '1',
    '--debug_steps', '1',
    '--database_path', 'H:\\PS_data',
    '--protocols_path', 'H:\\PS_data'
]

# 导入 main.py 的内容
import main

# 限制 DataLoader 只跑 1 个 batch，尽量降低资源占用
from torch.utils.data import DataLoader

original_iter = DataLoader.__iter__

def limited_iter(self):
    it = original_iter(self)
    for i, batch in enumerate(it):
        if i >= 1:
            break
        yield batch

DataLoader.__iter__ = limited_iter

# 运行 main
if __name__ == '__main__':
    exec(open('main.py').read())
