import os


def delete_logs_with_few_lines(folder_path, min_line_count, other_folder_path):
    """
    删除文件夹中行数少于指定数量的日志文件。

    :param folder_path: 包含日志文件的文件夹路径
    :param min_line_count: 最小行数要求
    :param other_folder_path: related folder path
    """
    log_file_names = set()

    # 遍历文件夹下的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.log'):  # 确保文件是以.log为后缀的日志文件
            file_path = os.path.join(folder_path, filename)

            # 打开文件并读取行数
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 检查行数是否小于最小行数
            if len(lines) < min_line_count:
                # 如果小于最小行数，则删除文件
                os.remove(file_path)
                print(f"Deleted: {file_path}")
                log_file_names.add(os.path.splitext(filename)[0])

    # 删除另一个文件夹中与日志文件名称相同的子文件夹及其内容
    for result_name in os.listdir(other_folder_path):
        result_path = os.path.join(other_folder_path, result_name)
        if os.path.isdir(result_path):
            if result_name in log_file_names:
                for root, dirs, files in os.walk(result_path, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        os.rmdir(dir_path)
                os.rmdir(result_path)
                print(f"已删除文件夹及其内容: {result_name}")


if __name__ == '__main__':
    delete_logs_with_few_lines('../log', 100, "../results")

