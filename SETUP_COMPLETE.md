# XLSR-Mamba 环境配置完成报告

## ✅ 已完成的修复

### 1. 依赖安装与兼容性
- ✅ Python 3.10.13
- ✅ PyTorch 2.1.1+cu118
- ✅ fairseq 1.0.0a0 (numpy 兼容性已修复)
- ✅ mamba-ssm 1.2.0.post1
- ✅ librosa 0.10.1
- ✅ numpy 1.26.3 (降级以兼容torch)
- ✅ transformers 4.38.1 (降级以兼容torch API)
- ✅ soundfile + libsndfile
- ✅ scipy, sklearn, pandas 等科学计算库

### 2. 代码修复
- ✅ `main.py`: 路径构建逻辑 (`prefix_2019.replace('.', '_')`)
- ✅ `main.py`: 添加所有 RawBoost 参数（N_f, nBands等21个参数）
- ✅ `main.py`: Windows文件移动 (`shutil.move` 替换 `os.system('mv')`)
- ✅ `data_utils.py`: 音频路径拼接 (`os.path.join` 替换字符串拼接)
- ✅ `fairseq`: numpy.float 废弃 API 修复
- ✅ DataLoader num_workers 改为 0 (Windows 兼容)

### 3. 数据结构验证
```
C:\Users\Administrator\Desktop\XLSR-Mamba-main\
  ├─ xlsr2_300m.pt                     ✅ XLSR-300M权重 (1.2GB)
  ├─ LA\
  │   ├─ ASVspoof2019_LA_cm_protocols\ ✅ 协议文件
  │   ├─ ASVspoof2019_LA_train\flac\   ✅ 20313个音频
  │   └─ ASVspoof2019_LA_dev\flac\     ✅ 19257个音频
  └─ models\                           ✅ 训练输出目录
```

---

## 🚀 运行方法

### 方案A：使用批处理脚本（推荐）
双击运行：
```
run_train.bat
```

### 方案B：手动命令行
在 **Anaconda Prompt (antimamba02)** 中：
```powershell
cd C:\Users\Administrator\Desktop\XLSR-Mamba-main
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
python main.py --algo 5 --batch_size 2 --num_epochs 1 --database_path . --protocols_path .
```

### 方案C：PowerShell
```powershell
cd C:\Users\Administrator\Desktop\XLSR-Mamba-main
conda activate antimamba02
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
python main.py --algo 5 --batch_size 2 --num_epochs 1 --database_path . --protocols_path .
```

---

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algo` | 5 | RawBoost算法(LA用5, DF用3) |
| `--batch_size` | 4 | 批次大小(显存不足改为2或1) |
| `--num_epochs` | 7 | 训练轮数(早停机制) |
| `--lr` | 1e-6 | 学习率 |
| `--emb_size` | 144 | 嵌入维度 |
| `--num_encoders` | 12 | Mamba层数 |

---

## 📊 预期输出

### 正常启动
```
Device: cuda
W2V + mamba
MambaConfig(d_model=144, n_layer=6, ...)
no. of training trials 25380
no. of validation trials 24844
######## Epoch 0 ########
0%|█████     | 1234/6345 [10:23<45:12, 1.5it/s]
```

### 训练完成后
- 模型保存：`models/Bmamba5_LA_WCE_1e-06_ES144_NE12/best/`
- 分数文件：`Scores/LA/*.txt`

---

## ⚠️ 常见问题

### 1. 显存不足 (CUDA OOM)
```powershell
python main.py --algo 5 --batch_size 1 --num_epochs 1 --database_path . --protocols_path .
```

### 2. CUDA_PATH 未设置
在运行前执行：
```powershell
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```

### 3. soundfile 找不到 DLL
```powershell
conda activate antimamba02
pip uninstall soundfile -y
pip install soundfile
```

### 4. 训练速度慢
- 原因：`num_workers=0` (主进程加载数据)
- 解决：Linux/WSL2 环境可改为 `num_workers=4`

---

## 📈 性能预期

- **1 epoch 时间**: 约30-60分钟 (RTX 3090/4090)
- **完整训练**: 7 epochs + 早停, 约3-7小时
- **EER目标** (ASVspoof2021 LA): 0.93% (论文结果)

---

## 🔍 下一步

1. 运行 `run_train.bat` 或手动命令
2. 观察前几个batch是否正常 (进度条开始滚动)
3. 若一切正常, 可改 `--num_epochs 7` 完整训练
4. 训练完成后运行评估：
   ```powershell
   python la_evaluate.py
   ```

---

## 📝 已修复的所有Bug

1. fairseq numpy.float 废弃
2. transformers torch API 不兼容
3. soundfile libsndfile.dll 缺失
4. main.py 路径拼接 (prefix_2019.split → replace)
5. main.py 路径重复 (_LA_LA_train)
6. data_utils.py 路径拼接 (字符串拼接 → os.path.join)
7. main.py 缺少 RawBoost 参数 (添加21个参数)
8. Windows mv 命令 (改为 shutil.move)
9. Triton CUDA_PATH 环境变量
10. DataLoader num_workers Windows兼容性

---

**生成时间**: 2025-11-06 00:30  
**环境**: antimamba02 (Python 3.10.13, PyTorch 2.1.1+cu118)  
**项目**: XLSR-Mamba (IEEE Signal Processing Letters)




