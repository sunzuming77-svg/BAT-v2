@echo off
echo ============================================
echo Installing mamba-ssm with CUDA extensions
echo ============================================

REM 激活 VS 2022 编译环境
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1

REM 设置 CUDA 环境
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_HOME=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"

REM 激活 conda 环境
call D:\code\anaconda\Scripts\activate.bat antimamba02 > nul 2>&1

echo.
echo [1/2] Installing causal-conv1d (this may take 5-10 minutes)...
D:\code\anaconda\envs\antimamba02\python.exe -m pip install causal-conv1d==1.1.1 --no-cache-dir

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install causal-conv1d
    pause
    exit /b 1
)

echo.
echo [2/2] Installing mamba-ssm (this may take 5-10 minutes)...
D:\code\anaconda\envs\antimamba02\python.exe -m pip install mamba-ssm==1.2.0.post1 --no-cache-dir

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install mamba-ssm
    pause
    exit /b 1
)

echo.
echo ============================================
echo Installation complete! Testing...
echo ============================================
D:\code\anaconda\envs\antimamba02\python.exe -c "import causal_conv1d; import mamba_ssm; print('SUCCESS: mamba-ssm', mamba_ssm.__version__)"

pause




