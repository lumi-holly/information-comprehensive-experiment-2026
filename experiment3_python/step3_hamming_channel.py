from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import (
    RESULT_DIR,
    bits_to_bytes,
    bytes_to_bits,
    ensure_result_dir,
    hamming74_decode,
    hamming74_encode,
    set_chinese_plot_style,
)


def run() -> dict[str, object]:
    ensure_result_dir()
    set_chinese_plot_style()
    input_file = RESULT_DIR / "aes_encrypted_data.npz"
    if not input_file.exists():
        raise FileNotFoundError("找不到第二步数据，请先运行 step2_aes_crypto.py。")

    cipher_bytes = np.load(input_file)["cipher_bytes"].astype(np.uint8)
    input_bits = bytes_to_bits(cipher_bytes)
    coded_matrix, padding_count = hamming74_encode(input_bits)
    code_rate = input_bits.size / coded_matrix.size
    rng = np.random.default_rng(2026)
    probabilities = np.array([0, 0.001, 0.005, 0.01, 0.02], dtype=float)
    ber_before = np.zeros(probabilities.size)
    ber_after = np.zeros(probabilities.size)
    corrected_counts = np.zeros(probabilities.size, dtype=int)

    padded_input = np.pad(input_bits, (0, padding_count), constant_values=False)
    print("===== 汉明(7,4)编码结果 =====")
    print(f"AES密文字节数：{cipher_bytes.size} bytes")
    print(f"AES密文比特数：{input_bits.size} bit")
    print(f"4位信息组数量：{coded_matrix.shape[0]}")
    print(f"编码后比特数：{coded_matrix.size} bit")
    print(f"码率：{code_rate:.4f}（理论值4/7={4/7:.4f}）")
    print("===== 随机误码信道仿真 =====")
    print("误码概率p\t译码前BER\t译码后BER\t执行纠错的码字数")
    for index, probability in enumerate(probabilities):
        error_mask = rng.random(coded_matrix.shape) < probability
        received = coded_matrix ^ error_mask
        decoded, _, syndrome = hamming74_decode(received)
        ber_before[index] = np.count_nonzero(error_mask) / coded_matrix.size
        ber_after[index] = np.count_nonzero(decoded != padded_input) / padded_input.size
        corrected_counts[index] = np.count_nonzero(syndrome)
        print(f"{probability:.3f}\t\t{ber_before[index]:.8f}\t{ber_after[index]:.8f}\t{corrected_counts[index]}")

    controlled = coded_matrix.copy()
    test_count = min(1000, coded_matrix.shape[0])
    selected_rows = rng.choice(coded_matrix.shape[0], size=test_count, replace=False)
    selected_positions = rng.integers(0, 7, size=test_count)
    controlled[selected_rows, selected_positions] ^= True
    controlled_decoded_padded, corrected_matrix, controlled_syndrome = hamming74_decode(controlled)
    controlled_decoded = controlled_decoded_padded[: input_bits.size]
    controlled_is_correct = np.array_equal(controlled_decoded, input_bits)
    recovered_cipher_bytes = bits_to_bytes(controlled_decoded)
    cipher_is_correct = np.array_equal(recovered_cipher_bytes, cipher_bytes)

    print("===== 单比特纠错验证 =====")
    print(f"主动加入错误：{test_count} bit")
    print(f"检测并执行纠错：{np.count_nonzero(controlled_syndrome)} 个码字")
    print(f"译码后与原始AES密文比特是否完全一致：{int(controlled_is_correct)}")
    print(f"恢复的AES密文字节是否完全正确：{int(cipher_is_correct)}")
    if not controlled_is_correct or not cipher_is_correct:
        raise RuntimeError("汉明码单比特纠错验证失败。")

    np.savez_compressed(
        RESULT_DIR / "hamming_decoded_data.npz",
        recovered_cipher_bytes=recovered_cipher_bytes,
        probabilities=probabilities,
        ber_before=ber_before,
        ber_after=ber_after,
        corrected_counts=corrected_counts,
        code_rate=code_rate,
        original_bit_count=input_bits.size,
        padding_bit_count=padding_count,
        controlled_is_correct=controlled_is_correct,
    )

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.semilogy(probabilities[1:], ber_before[1:], "o-b", label="译码前BER")
    axis.semilogy(probabilities[1:], np.maximum(ber_after[1:], 1e-8), "s-r", label="译码后BER")
    axis.set(title="汉明(7,4)译码前后误码率对比", xlabel="信道误码概率 p", ylabel="误码率 BER")
    axis.grid(True, which="both", alpha=0.35)
    axis.legend(loc="upper left")
    fig.savefig(RESULT_DIR / "04_汉明码误码率曲线.png", dpi=180)
    plt.close(fig)

    example_row = int(selected_rows[0])
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), constrained_layout=True)
    series = [coded_matrix[example_row], controlled[example_row], corrected_matrix[example_row]]
    titles = [
        "发送端正确码字",
        f"信道接收码字：第{selected_positions[0] + 1}位发生错误",
        "汉明译码纠错后的码字",
    ]
    for axis, values, title in zip(axes, series, titles):
        axis.stem(np.arange(1, 8), values.astype(int))
        axis.set(title=title, ylabel="比特值", ylim=(-0.2, 1.2))
        axis.grid(True, alpha=0.35)
    axes[-1].set_xlabel("码字位置")
    fig.savefig(RESULT_DIR / "05_汉明码单比特纠错示例.png", dpi=180)
    plt.close(fig)

    print("第三步汉明信道编码与译码完成。\n")
    return {
        "probabilities": probabilities,
        "ber_before": ber_before,
        "ber_after": ber_after,
        "controlled_is_correct": controlled_is_correct,
        "cipher_is_correct": cipher_is_correct,
    }


if __name__ == "__main__":
    run()

