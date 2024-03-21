import os
def find_image(file_path, target_character,folder_path):
    target_number = None
    with open(file_path, 'r') as file:
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
#find_image("numbers.txt","皑","good")