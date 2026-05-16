import csv
from pathlib import Path

def scan_csv_for_invalid_chars(file_path):
    """
    扫描 CSV 文件中包含非 UTF-8 或非英文可打印字符的行
    只报告，不修改
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    print(f"🔍 开始扫描文件: {file_path}")
    print("-" * 80)

    invalid_lines = []
    allowed_bytes = set(range(32, 127))  # 可打印 ASCII: 字母、数字、标点、空格
    allowed_bytes.update([9, 10, 13])    # \t, \n, \r

    with open(file_path, 'rb') as f:
        lines = f.readlines()

    for line_num, raw_line in enumerate(lines, 1):
        has_invalid = False
        bad_bytes = []

        for byte in raw_line:
            if byte not in allowed_bytes:
                has_invalid = True
                bad_bytes.append(byte)

        if has_invalid:
            try:
                text_preview = raw_line.decode('utf-8').strip()
            except:
                text_preview = raw_line.decode('utf-8', errors='replace').strip()

            invalid_lines.append({
                'line': line_num,
                'preview': text_preview[:100] + ("..." if len(text_preview) > 100 else ""),
                'bad_bytes_hex': ' '.join(f"{b:02X}" for b in bad_bytes[:10]) + (" ..." if len(bad_bytes) > 10 else "")
            })

    # === 输出结果 ===
    if invalid_lines:
        print(f"🚩 发现 {len(invalid_lines)} 行包含非法字符：")
        print("-" * 80)
        for item in invalid_lines:
            print(f"行 {item['line']:>4}: {item['preview']}")
            print(f"         非法字节 (十六进制): {item['bad_bytes_hex']}")
            print()
        print("💡 建议：")
        print("   1. 用 Notepad++ 或 VS Code 打开该文件")
        print("   2. 跳转到上述行号")
        print("   3. 删除或替换乱码部分为普通空格")
        print("   4. 保存为 UTF-8 编码")
    else:
        print("✅ 所有行均符合英文 ASCII 标准，无非法字符！")

# === 执行 ===
file_path = r'D:/#第一个喵喵/NFRKG_20251104/NFRKG/pybert/dataset/review_origin_train.csv'
scan_csv_for_invalid_chars(file_path)