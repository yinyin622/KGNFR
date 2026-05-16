import os
import csv

def process_csv(input_path, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    with open(input_path, mode='r', encoding='utf-8') as infile, \
         open(output_path, mode='w', encoding='utf-8', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 读取表头
        header = next(reader)
        # 预期原始表头: ['id', 'review', 'Usa', 'Sup', 'Dep', 'Per']
        if len(header) != 6:
            raise ValueError(f"Expected 6 columns in header, got {len(header)}: {header}")

        # 新表头：将 'Dep' 替换为 'Rel'，并添加 'Mis'
        new_header = ['id', 'review', 'Usa', 'Sup', 'Rel', 'Per', 'Mis']
        writer.writerow(new_header)

        for row in reader:
            if len(row) != 6:
                print(f"Warning: Skipping malformed row (expected 6 cols): {row}")
                continue

            id_val = row[0]
            review = row[1]
            # 原始标签：Usa, Sup, Dep(→Rel), Per
            try:
                usa = int(row[2])
                sup = int(row[3])
                dep = int(row[4])   # this is actually 'Rel'
                per = int(row[5])
            except ValueError as e:
                print(f"Warning: Invalid label in row {row}, skipping. Error: {e}")
                continue

            labels_4 = [usa, sup, dep, per]
            if sum(labels_4) == 0:
                mis = 1
            else:
                mis = 0

            new_row = [id_val, review, usa, sup, dep, per, mis]
            writer.writerow(new_row)

    print(f"✅ Processed {input_path} → {output_path}")



if __name__ == "__main__":
    input_file = "D:/#第一个喵喵/NFRKG_20251104/NFRKG/pybert/dataset/TTA_ALL_TEST.csv"
    output_dir = "D:/#第一个喵喵/NFRKG_20251104/NFRKG/pybert/dataset/Mis"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    process_csv(input_file, output_dir)