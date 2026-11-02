# 实验五：完整语音处理与安全网络通信系统

本工程将前面实验串联为可运行的 Python 程序：

```text
Windows 发送端
WAV / 麦克风 → 保存参考语音、时频分析 → 加噪
→ DPCM8 / PCM16 → AES-128-CBC + HMAC → Hamming(7,4) / 不加信道码
                         ↓ 网线或 Wi-Fi，真实 TCP 网络
RK3399 Ubuntu 接收端
信道译码 → 认证与解密 → 信源解码 → 连续 FIR 带通滤波
→ 保存 WAV / 可选播放 → 时频分析、质量与性能评价
```

正式运行使用两台设备，接收端先启动。无需先做本机回环通信。麦克风和扬声器为可选硬件路径，使用 WAV 文件即可验证文件输入的完整处理链。PyQt 是课件选做内容，本工程提供命令行界面。

课件要求、参数来源和未实现的扩展项见 [COURSE_REQUIREMENTS.md](COURSE_REQUIREMENTS.md)。本工程不预写 RK3399、麦克风或真实网络的实测结论，实际运行后以输出数据为准。

## 1. 文件与依赖

| 文件 | 作用 |
| --- | --- |
| `app.py` | 统一入口：密钥生成、发送、接收、质量评价、结果汇总、音频设备查询 |
| `audio_dsp.py` | WAV 读写、连续加噪、自编带通 FIR、时频数据、MSE/SNR |
| `audio_capture.py` | 后台连续录音和有界队列，避免网络等待直接阻塞采集 |
| `codec.py` | DPCM8/PCM16、AES-CBC/HMAC、汉明编解码、比特打包、模拟误码 |
| `network_io.py` | TCP 长度分帧、完整接收、认证控制消息 |
| `requirements.txt` | 文件/算法/网络模式依赖 |
| `requirements-audio.txt` | 麦克风与播放的额外依赖 |
| `tests/` | 算法和协议的软件测试；测试通过不等于板端硬件已测 |

程序使用 Python 3.8+，基础依赖为 NumPy 和 cryptography，不需要 MATLAB、SciPy 或开发板专有 SDK。RK3399 必须使用支持当前依赖的 Python 环境；如果板载 Python 过旧，先安装较新 Python，不能直接假定系统自带版本可用。

Windows 克隆或下载本仓库后，在 PowerShell 中进入本目录：

```powershell
cd experiment5_system
python -m pip install -r requirements.txt
python app.py --help
```

把整个 `experiment5_system` 文件夹复制到 RK3399，在 Ubuntu 终端进入复制后的文件夹，再执行：

```bash
python3 --version
python3 -m pip install -r requirements.txt
python3 app.py --help
```

下文 Windows 示例用 `python`，Ubuntu 示例用 `python3`。每条命令都在本程序目录中执行。如果使用虚拟环境，先激活对应虚拟环境。

## 2. 准备网络和密钥

1. Windows 与 RK3399 连接到能互通的同一局域网，可通过路由器的网线或 Wi-Fi。使用网线直连时，需要自行配置同网段地址。
2. 在 RK3399 终端运行 `hostname -I`，找到实际使用网卡的 IPv4 地址。下面的 `192.168.1.100` 仅为示例，需要替换。
3. 在 Windows 只生成一次密钥：

```powershell
python app.py keygen --out shared.key
```

4. 通过可信方式把同一个 `shared.key` 文件复制到 RK3399 的程序目录。两端各自重新生成的密钥不同，不能通信。程序不会打印密钥内容，也不会覆盖已有密钥文件。

`shared.key` 是本实验共享的 48 字节二进制文件，分别用于 AES 和 HMAC。不要改成文本，不要随实验源代码公开上传。源代码的 `.gitignore` 已排除密钥及运行结果。

若 Windows 作为接收端，允许 Python 通过所用专用网络的防火墙；若 Ubuntu 启用了防火墙，需要允许所用 TCP 端口。`0.0.0.0` 是接收端监听地址，发送端 `--host` 必须填写接收设备实际 IP。

## 3. 第一次完整传输：Windows WAV → RK3399

先在 RK3399 上运行：

```bash
python3 app.py receive --bind 0.0.0.0 --port 5000 --key shared.key --out results/rk3399
```

看到“等待发送端连接”后，在 Windows 运行：

```powershell
python app.py send --host 192.168.1.100 --port 5000 --key shared.key --wav original.wav --out results/windows
```

本公开仓库只发布代码，不附带个人录音。请把实验一的同一份原始语音复制到 `experiment5_system/original.wav`，或通过 `--wav` 指定自己的 16 bit PCM WAV 路径；也可使用下一节的麦克风模式。立体声输入会先按声道平均转为单声道；输入 WAV 必须为未压缩的 16 bit PCM。

默认配置为：DPCM8、AES-128-CBC/HMAC、Hamming(7,4)、5000 Hz/幅值 0.2 正弦干扰、300～3400 Hz 的 129 tap FIR。文件模式读取 WAV 实际采样率；`--sample-rate` 只控制麦克风。

本课程原始语音为 16 kHz。若使用 8 kHz 等其他 WAV，默认 5 kHz 干扰不满足采样定理，程序会拒绝运行。首次实验建议继续使用同一份 16 kHz 原始 WAV，方便和实验三比较。

**完成标志：** 两端都显示完成，两个 summary JSON 均为 `"status": "complete"`，帧数和样本数匹配，接收端生成 `recovered.wav`。只有部分帧文件而没有完整结束确认时，本次运行不算成功。

接收程序只接收一次会话后退出。每次重新实验，都要重新启动接收程序，并为两端指定新的 `--out` 目录。程序发现结果目录已有 WAV 或运行摘要时会拒绝覆盖，保护已有实验数据。

## 4. 麦克风采集和边接收边播放

只做 WAV 传输不需要本节依赖。要使用麦克风或播放，在需要音频硬件的设备上执行：

```bash
python3 -m pip install -r requirements-audio.txt
```

Windows 将上面的 `python3` 换为 `python`。Ubuntu 如果缺少 PortAudio，在能够安装系统软件的环境中执行：

```bash
sudo apt install libportaudio2
```

若 pip 需要从源码构建音频依赖，还可能需要 `portaudio19-dev` 和编译环境。是否可录音/播放取决于开发板实际音频接口、驱动和麦克风/扬声器，不把安装依赖等同于硬件验证。

查询设备：

```powershell
python app.py audio-devices
```

接收端可选播放：

```bash
python3 app.py receive --key shared.key --out results/rk3399_mic --play
```

Windows 实时采集 5 秒并发送：

```powershell
python app.py send --host 192.168.1.100 --key shared.key --mic --seconds 5 --sample-rate 16000 --out results/windows_mic
```

可用 `--input-device 1`、接收端 `--output-device 2` 选择设备，编号以各自 `audio-devices` 的输出为准。

麦克风在后台线程连续采集，通过默认 16 帧有界队列交给处理和发送线程，减少网络等待对采集的影响。容量是实现建议值，不是课件阈值。程序对每帧执行处理、发送和确认；接收端播放也会占用时间。若输入溢出或队列满，本次运行明确失败，不静默丢样本。可先关闭接收端 `--play` 验证采集与保存，再根据记录调整 `--block-samples`；不能事先保证任意硬件上无溢出。

文件模式要模拟录音的原始时间节奏，可在发送命令末尾加 `--pace`。不加此项时尽快传输，适用于吞吐与处理能力测试。`--pace` 不能保证系统达到实时要求，如果处理本身过慢仍会落后。

## 5. 不同条件下比较编解码方案

每一组都使用同一 WAV、同一 `--headroom`、帧长、加噪与滤波参数；先固定 `--channel-test none`。每次更换方案前重启接收端，并使用新的输出目录。

| 实验目的 | 发送端参数 | 重点记录 |
| --- | --- | --- |
| 复现实验三的主要编码组合 | `--source dpcm8 --channel hamming74` | 分模块耗时、实际帧字节数、恢复音质 |
| 信源编码对照 | `--source pcm16 --channel hamming74` | 同样输入下 PCM16 与 DPCM8 的音质和数据量取舍 |
| 信道编码对照 | `--source dpcm8 --channel none` | 汉明码增加的冗余和处理耗时 |
| 排除人工干扰 | `--tone-amplitude 0` | 不加人工正弦噪声时，编码与滤波本身对波形的影响 |

例如 PCM16 对照组，接收端先用新的目录启动，发送端运行：

```powershell
python app.py send --host 192.168.1.100 --key shared.key --wav original.wav --source pcm16 --channel hamming74 --out results/windows_pcm16
```

这是手动选择模式，没有未经测量的自动切换阈值。课件提到的 APCM/ADPCM、Lloyds、`dpcmopt` 未在本工程实现；当前已实现的是 PCM16、DPCM8 和汉明码对照。

### 比特打包与计算优化

汉明编码输出按位打包后发送，不把每个 0/1 作为一个字节或字符发送。汉明译码采用 NumPy 批量计算校验子。程序没有额外实现一个逐码字循环的“慢版本”，因此不要仅凭源码结构填写“提速若干倍”；如需量化提速，必须另设等价基准并实测。

降低量化位数但仍按每样本一个字节保存，并不会减少发送字节。因此本工程不提供一个没有配套打包的“6 bit DPCM 压缩”选项。

## 6. 单比特纠错与随机误码实验

TCP 不会由本程序主动制造网络误码。以下两种模式是应用程序在发送前主动翻转编码比特的模拟实验，结果必须标记为“模拟误码”，不能写成真实局域网测得的 BER。

受控单比特错误：每帧至多选前 16 个汉明码字，每个码字翻转一位：

```powershell
python app.py send --host 192.168.1.100 --key shared.key --wav original.wav --channel hamming74 --channel-test single --out results/windows_single
```

随机比特错误示例：

```powershell
python app.py send --host 192.168.1.100 --key shared.key --wav original.wav --channel hamming74 --channel-test random --ber 0.001 --out results/windows_random
```

`0.001` 是实验配置示例，不是课件指定标准。程序 BER 范围 0～0.02 是实施范围，不是老师规定的上限。随机误码可能在同一码字内产生多个错误，汉明码不能保证全部恢复；认证失败时本次传输明确失败，部分输出不能当成完整语音。

`syndrome_codewords` 记录非零校验子的码字数，表示执行译码修正的次数。不能仅凭这个数称“所有错误已纠正”；还要检查最终认证和完整结束确认。

## 7. 质量评价

把接收端的结果目录复制回 Windows，例如放到本程序目录的 `results/rk3399`。发送端的 `reference.wav` 是乘以默认 `headroom=0.75` 后、加入人工噪声前的参考语音；应使用它作为基准，不能直接用幅值不同的原始 WAV。

```powershell
python app.py evaluate --reference results/windows/reference.wav --received results/rk3399/recovered.wav --decoded results/rk3399/decoded_noisy.wav --delay-samples 64 --out results/evaluation.json
```

输出包含：

- `against_clean_delay_aligned`：接收滤波结果与干净参考比较，补偿指定延迟后的 MSE/SNR。
- `against_same_filter_clean`：干净参考通过同一 FIR 后再比较，用于区分滤波改变频带与编码/传输误差。
- `decoded_noisy_against_clean`：接收端滤波前的解码信号与干净参考比较，仅在提供 `--decoded` 时计算。

默认 129 tap 线性相位 FIR 固定延迟为 64 样本，16 kHz 下为 4 ms。若修改发送端 `--taps`、`--low-hz`、`--high-hz`，评价时也要使用相同参数，并设置 `--delay-samples (taps-1)/2` 的整数结果。`--taps 0` 可关闭“同滤波参考”项，不会自动关闭延迟补偿。

程序要求采样率与总样本数一致，延迟补偿后须仍有样本。完全一致或参考为静音时，SNR 可能没有有限值，JSON 使用 `null` 并给出 `snr_status`，不是计算出了 0 dB。

评价命令会检查 WAV 同目录的 `send_summary.json` 与 `receive_summary.json`，任一为 `incomplete` 就拒绝评价。没有相邻摘要时，只进行离线文件评价，并在结果标明缺少完整运行证据。

## 8. 性能与平台比较

```powershell
python app.py compare --runs results/windows/send_summary.json results/rk3399/receive_summary.json --out results/comparison.csv
```

此命令汇总已有结果，不自动判断谁更快。主要记录包括：

| 字段 | 正确含义 |
| --- | --- |
| 各模块 `*_ms` | 对应阶段实际测量耗时，具体见逐帧 CSV；不要把重叠计时字段相加当总耗时 |
| `turnaround_ms` | 发送一帧到收到确认，包含接收端译码、滤波、保存与可选播放，不能写成纯网络单程延迟 |
| `recv_wait_ms` | 接收等待，包含发送端采集/处理/节奏等待，不等于网络传播时延 |
| `audio_framed_bytes` | 音频消息及应用层长度/类型封装字节数，不包含全部控制消息、ACK 或 TCP/IP 链路开销 |
| `effective_session_bps` | 本次会话时间内上述字节数换算的有效速率，不等于网卡带宽或纯网络极限吞吐 |
| `over_budget_frames` | 发送端活动处理与等待超过该帧音频时长的帧数；默认帧长 1024 在 16 kHz 下为 64 ms |
| `playback_underflows` | 播放缓冲不足次数，需结合听感和硬件实际检查 |

要比较 Windows 与 RK3399 的**相同模块**处理性能，需要交换角色再运行一次：

1. 第一轮 Windows 发送、RK3399 接收，保存两端结果。
2. 第二轮 Windows 先启动 `receive`，RK3399 用同一 WAV 和配置执行 `send`。第二轮发送端 IP 改为 Windows 实际 IP。
3. 用两轮的发送端编码耗时比较编码性能，用两轮接收端译码/滤波耗时比较接收处理性能。不要拿 Windows 编码与 RK3399 译码的总时间直接判定平台快慢。
4. 文件模式下重复若干次，保持 `--pace`、`--play`、帧长和运行负载一致，报告平均值与波动，并保留原始记录。

与实验三 MATLAB 比较时，固定输入、幅值、算法及参数，并单独记录处理时间。网络版 DPCM 每帧独立预测，与实验三整段预测在帧边界有差异；FIR 与原 MATLAB 的 Butterworth/filtfilt 也不同。若需严格逐样本对照，必须先统一这些条件。真实网络总时间不能直接当作 MATLAB/Python 的算法速度差异。

## 9. 每次运行保存什么

发送端输出：`reference.wav`、`noisy.wav`、`send_frames.csv`、`send_summary.json`，以及各语音的时域与频谱 CSV。

接收端输出：`decoded_noisy.wav`、`recovered.wav`、`receive_frames.csv`、`receive_summary.json`，以及各语音的时域与频谱 CSV。

时间 CSV 为最多约 4000 个显示点，并标明抽取步长；频谱最多分析前 32768 点，文件中标明是否使用前缀子集。质量 MSE/SNR 则使用评价命令中对齐后的有效样本。可用 Excel 或 MATLAB 绘制 CSV，不把抽取的显示点当作处理时丢弃的样本。

报告建议保留：两端终端完成截图、网络拓扑和实际设备信息、运行命令、完整 JSON/CSV、相同时间段的波形与频谱、恢复语音，以及明确的未测项。程序报错留下的 `incomplete` 结果只能作为故障证据。

## 10. 常见问题

| 现象 | 检查或处理 |
| --- | --- |
| 无法连接/连接超时 | 接收端是否先启动、实际 IP 和端口、同网段互通、防火墙；默认等待 60 秒可用 `--timeout` 调整 |
| 认证失败 | 两端是否使用同一个未修改的密钥；若开启随机误码，是否出现超出汉明纠错能力的错误 |
| 加噪频率超范围 | 使用 16 kHz 原始语音，或合理调整干扰和滤波参数；必须符合采样定理 |
| headroom 与幅值不合法 | `headroom + tone-amplitude` 不得超过 1，避免人工加噪后削波；默认 0.75+0.2=0.95 |
| 音频依赖或设备失败 | 安装可选音频依赖，查询对应设备，检查麦克风权限、音频驱动和物理连接 |
| 麦克风溢出或播放断续 | 查看逐帧耗时，先关闭 `--play`，检查设备是否支持 16 kHz，再按实际情况调整帧长 |
| 有 WAV 但 summary 为 incomplete | 传输中断或校验失败，不能作为完整结果；保存故障记录后重新运行到新目录 |

## 11. 软件验证与实际验收的区别

软件测试命令：

```powershell
python -m unittest discover -s tests -v
```

测试中的本机连接只用于验证协议和异常处理，不是正式实验必做步骤，也不代表真实网络性能。Windows↔RK3399、麦克风与扬声器、不同平台处理速度和真实网络结果，均需要在实际设备上执行后再填写结论。具体本次代码验证范围见 [TEST_RESULTS.md](TEST_RESULTS.md)。

## 12. 实现参考

- [Python Socket Programming HOWTO](https://docs.python.org/3/howto/sockets.html)：TCP 为字节流，需要长度封装和循环接收。
- [cryptography 对称加密文档](https://cryptography.io/en/latest/hazmat/primitives/symmetric-encryption/)：CBC 加密需结合认证，使用随机 IV，先验证消息再接受解密结果。
- [NumPy packbits](https://numpy.org/doc/stable/reference/generated/numpy.packbits.html)：将比特打包为实际字节。
- [sounddevice 官方接口](https://python-sounddevice.readthedocs.io/en/latest/api/index.html)：音频设备、阻塞式采集/播放及溢出状态。
