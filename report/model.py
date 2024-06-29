import os
import time

def getResult(image_file):
    print(image_file)
    print("开始处理图片........")
    time.sleep(10)
    print("处理完成........")
    # 创建json对象
    result = {
        "image_file": "你好",
        "result": "positive"
    }
    return result
