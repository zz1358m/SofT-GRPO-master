import json
import argparse
import re
def extract_code( text: str) -> str:
    # Use regex to find the content inside ```python\n...\n```
    matches = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    # Return the last match if any matches exist
    completion_code = matches[-1] if matches else ""
    return completion_code


def convert_json(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    # 初始化结果列表
    result = []

    # 遍历数据并处理
    for item in data:
        question_id = item.get("ground_truth").get("question_id")
        completion = item.get("completion")

        # 如果 completion 是字符串，转换为单元素列表
        if isinstance(completion, str):
            completion = [extract_code(completion)]
        else:
            completion = [extract_code(c) for c in completion]

        # 构造新的字典
        result.append({"question_id": question_id, "code_list": completion})

    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(result, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Convert JSON file format for LiveCodeBench.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSON file.")
    # parser.add_argument("--file_type", type=str, help="Type of the file (e.g., baseline or ours).")

    args = parser.parse_args()

    # 调用转换函数
    convert_json(args.input_file, args.output_file)