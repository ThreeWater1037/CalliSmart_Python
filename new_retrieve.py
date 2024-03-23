import os
# 定义一个函数find_image，用于根据文件路径、目标字符和文件夹路径查找图片

def find_image(file_path, target_character,folder_path):
    target_number = None
    # 定义一个变量target_number，用于存储目标字符对应的数字
    with open(file_path, 'r') as file:
        # 打开文件，读取每一行
        for line in file:
            chinese_character = line.strip()[:len(target_character)]
            number = line.strip()[len(target_character):]
            if chinese_character == target_character:
                target_number = number
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith(target_number) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_path = os.path.join(root, file)
                print(image_path)
                return image_path
    return None
#下面为测试代码
#find_image("numbers.txt","皑","good")