import base64

def image_to_bytes(image_path):
    with open(image_path, 'rb') as image_file:
        image_bytes=image_file.read()
        image_str=base64.b64encode(image_bytes).decode('utf-8')
    return image_str

image_str=image_to_bytes("1.jpg")
print(image_str)