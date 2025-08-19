import random
import os

def generate_addition_data(filename, num_samples=10000, min_digits_a=3, max_digits_a=5, min_digits_b=3, max_digits_b=5):
    samples = []

    for _ in range(num_samples):
        digits_a = random.randint(min_digits_a, max_digits_a)
        digits_b = random.randint(min_digits_b, max_digits_b)

        a = random.randint(10**(digits_a - 1), 10**digits_a - 1)
        b = random.randint(10**(digits_b - 1), 10**digits_b - 1)

        input_seq = f"{a}+{b}"
        target_seq = str(a + b)

        samples.append((input_seq, target_seq))

    with open(filename, 'w', encoding='utf-8') as f:
        for inp, out in samples:
            f.write(f"{inp}\t{out}\n")

    print(f"生成完成，共 {num_samples} 条样本，保存在 {filename}")

if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)

    # 生成训练集
    generate_addition_data("data/addition_train.tsv", num_samples=10000)

    # 生成验证集（用不同数量和随机样本）
    generate_addition_data("data/addition_val.tsv", num_samples=2000)
