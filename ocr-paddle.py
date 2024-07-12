from paddleocr import PaddleOCR, draw_ocr
from ast import literal_eval
import numpy as np
from PIL import Image
import json
import time

def ocr(image):
    start_time = time.time()
    paddleocr = PaddleOCR(lang="en",ocr_version="PP-OCRv4",show_log = False,use_gpu=False)

    book_image = Image.open(image)
    book_image_array = np.array(book_image.convert('RGB'))

    result = paddleocr.ocr(book_image_array,cls=True)
    result = result[0]
    boxes = [line[0] for line in result]       #boundign box
    txts = [line[1][0] for line in result]     #raw text
    scores = [line[1][1] for line in result]   # scores
    # return  txts, result
    end_time = time.time()
    string_data = ' '.join(txts)
    # print(50*"--","\ntext only:\n",string_data)
    # print(50*"--","\nocr boxes:\n",result)

    # return {
    #     "text": string_data,
    #     "text_arr": txts,
    #     # "boxes": result
    # }
    elapsed_time = (end_time - start_time) * 1000
    return json.dumps({
        "text": string_data,
        "text_arr": txts,
        "time": elapsed_time,
        # "boxes": result
    })

# perform ocr scan
# local_book_image='17- (8).jpg'

print(ocr('17- (1).jpg'))
# print(ocr('/ringkasan.jpeg'))


