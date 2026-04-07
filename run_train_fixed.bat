@echo off
REM XLSR-Mamba Training Script - Fixed Version
call D:\code\anaconda\Scripts\activate.bat antimamba02
python main.py --algo 5 --batch_size 2 --num_epochs 1 --database_path . --protocols_path .
pause


