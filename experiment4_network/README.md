# 实验四 Python 网络语音通信

本工程完成实验四要求的通信链路：

`语音 -> DPCM -> AES-128-CBC -> Hamming(7,4) -> TCP -> Hamming译码 -> AES解密 -> DPCM解码 -> 语音`

## 1. 安装依赖

Windows 和 RK3399 的 Ubuntu 均执行：

```bash
python3 -m pip install -r requirements.txt
```

需要使用麦克风或实时播放时，再安装：

```bash
python3 -m pip install sounddevice
```

RK3399 如果安装 `sounddevice` 报错，先执行：

```bash
sudo apt update
sudo apt install -y portaudio19-dev libsndfile1
```

## 2. 先做本机回环测试

```bash
python3 loopback_demo.py
```

看到“回环测试通过”并生成 `loopback_received.wav`，说明编解码和 TCP 传输链路正常。

## 3. 两台设备传输 WAV

先在接收端运行：

```bash
python3 receiver.py --host 0.0.0.0 --port 5000 --output received_audio.wav
```

再在发送端运行，把 `192.168.1.100` 替换为接收端实际 IP：

```bash
python3 sender.py --host 192.168.1.100 --port 5000 --wav original.wav --realtime
```

两台设备必须在同一局域网。Windows 首次运行接收端时，需要允许 Python 通过专用网络防火墙。

## 4. 麦克风实时采集

接收端：

```bash
python3 receiver.py --host 0.0.0.0 --port 5000 --output mic_received.wav --play
```

发送端采集 10 秒：

```bash
python3 sender.py --host 192.168.1.100 --port 5000 --mic --seconds 10
```

## 5. 误码实验

正常 TCP 传输不会主动产生误码。为观察汉明码纠错，可以在发送端加入较低误码率：

```bash
python3 sender.py --host 192.168.1.100 --wav original.wav --ber 0.0001
```

如果同一个汉明码字出现两个或更多错误，程序会通过 SHA-256 校验发现该帧损坏并停止，避免输出无法识别的语音。

## 文件说明

- `signal_chain.py`：DPCM、AES、汉明码及网络帧封装。
- `network_io.py`：TCP 变长帧收发。
- `sender.py`：WAV 或麦克风发送端。
- `receiver.py`：接收、解码、保存或播放语音。
- `loopback_demo.py`：单机自动测试，不需要开发板。

AES 密钥目前写在教学代码中，便于两端直接复现实验。真实通信系统应通过安全方式分发密钥，不能把密钥提交到公开仓库。

