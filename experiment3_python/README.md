# 实验三 Python 版本

处理链路：`original.wav -> DPCM -> AES-128-CBC -> Hamming(7,4) -> 信道 -> 译码 -> 解密 -> DPCM解码`。

## 在 VS Code 中运行

1. 用 VS Code 打开 `python_version` 文件夹。
2. 按 `Ctrl+Shift+P`，选择 `Python: Select Interpreter`，选择已安装依赖的 Python。
3. 在终端执行：

```powershell
python -m pip install -r requirements.txt
python main_exp3.py
```

也可以打开“运行和调试”，选择“实验三：完整流程”后按 `F5`。

## 分步骤运行

```powershell
python step1_source_coding.py
python step2_aes_crypto.py
python step3_hamming_channel.py
```

必须按上述顺序执行。所有数据、图片和恢复语音保存在 `python_results` 中，不会覆盖 MATLAB 的 `results`。

## 准备语音样本

把一段 WAV 语音命名为 `original.wav`，放在本目录中。仓库没有上传实验语音和运行结果。

也可以通过环境变量指定其他位置：

```powershell
$env:EXPERIMENT_AUDIO = "D:\audio\original.wav"
python main_exp3.py
```

## 说明

- 为兼容原实验目录，如果本目录没有 `original.wav`，程序也会尝试读取 `../../实验一/original.wav`。
- DPCM 的量化位数、步长和预测方式与 MATLAB 版本相同。
- AES 使用固定 IV 只是为了复现实验；真实系统应使用随机且不重复的 IV。
- Python 与 MATLAB 的随机数发生器不同，因此随机信道的具体 BER 可能略有差异，但变化趋势和受控单比特纠错结果应一致。

