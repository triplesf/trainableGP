import os


def delete_logs_with_few_lines(folder_path, min_line_count):
    """
    删除文件夹中行数少于指定数量的日志文件。

    :param folder_path: 包含日志文件的文件夹路径
    :param min_line_count: 最小行数要求
    """
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


if __name__ == '__main__':
    # 调用函数，传入文件夹路径和最小行数
    folder_path = '../log'  # 替换成你的文件夹路径
    min_line_count = 100  # 设置最小行数
    delete_logs_with_few_lines(folder_path, min_line_count)

