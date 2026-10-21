from __future__ import annotations

import numpy as np
from scipy.io import wavfile

import step1_source_coding
import step2_aes_crypto
import step3_hamming_channel
from common import RESULT_DIR, aes_decrypt, dpcm_decode


def main() -> None:
    print("\n========== 实验三 Python 完整流程 ==========\n")
    step1_source_coding.run()
    step2_aes_crypto.run()
    step3_hamming_channel.run()

    source_data = np.load(RESULT_DIR / "dpcm_encoded_data.npz")
    channel_data = np.load(RESULT_DIR / "hamming_decoded_data.npz")
    recovered_cipher = channel_data["recovered_cipher_bytes"].astype(np.uint8)
    recovered_dpcm = np.frombuffer(aes_decrypt(recovered_cipher.tobytes()), dtype=np.uint8).copy()
    original_dpcm = source_data["src_bytes"].astype(np.uint8)
    if not np.array_equal(recovered_dpcm, original_dpcm):
        raise RuntimeError("完整链路失败：接收端 AES 解密数据与发送端 DPCM 数据不一致。")

    final_audio = dpcm_decode(recovered_dpcm, float(source_data["delta"]))
    reference_audio = source_data["decoded_audio"].astype(np.float64)
    final_consistent = np.array_equal(final_audio, reference_audio)
    fs = int(source_data["fs"])
    wavfile.write(
        RESULT_DIR / "final_received_audio.wav",
        fs,
        np.round(np.clip(final_audio, -1, 1) * 32767).astype(np.int16),
    )

    print("========== 完整链路验收 ==========")
    print("发送：语音 -> DPCM -> AES -> 汉明码 -> 信道")
    print("接收：汉明译码 -> AES解密 -> DPCM解码 -> 语音")
    print(f"接收端DPCM字节完全一致：{int(np.array_equal(recovered_dpcm, original_dpcm))}")
    print(f"最终语音与第一步DPCM解码结果完全一致：{int(final_consistent)}")
    print(f"最终接收语音：{RESULT_DIR / 'final_received_audio.wav'}")
    print("实验三 Python 版本运行成功。")


if __name__ == "__main__":
    main()

