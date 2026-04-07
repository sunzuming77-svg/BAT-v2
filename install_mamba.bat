@echo off
REM 配置 Visual Studio 2022 Build Tools 环境
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM 设置 CUDA 路径
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set CUDA_HOME=%CUDA_PATH%

REM 激活 conda 环境
call D:\code\anaconda\Scripts\activate.bat antimamba02

REM 安装 causal-conv1d
echo Installing causal-conv1d...
python -m pip install causal-conv1d==1.1.1 --no-cache-dir

REM 安装 mamba-ssm
echo Installing mamba-ssm...
python -m pip install mamba-ssm==1.2.0.post1 --no-cache-dir

echo.
echo Installation complete!
pause




