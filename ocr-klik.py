import cv2
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import json
import time

def ocr_from_frame(frame, ocr_model):
    start_time = time.time()
    
    # Konversi frame OpenCV ke format yang dapat dibaca oleh PaddleOCR
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    result = ocr_model.ocr(np.array(pil_image), cls=True)
    
    # Pastikan hasil OCR tidak kosong
    if not result or not result[0]:
        return json.dumps({
            "text": "",
            "text_arr": [],
            "time": (time.time() - start_time) * 1000,
        })
    
    result = result[0]
    txts = [line[1][0] for line in result]  # Raw text
    
    end_time = time.time()
    string_data = ' '.join(txts)
    elapsed_time = (end_time - start_time) * 1000
    
    return json.dumps({
        "text": string_data,
        "text_arr": txts,
        "time": elapsed_time,
    })

def main():
    # Inisialisasi model OCR
    ocr_model = PaddleOCR(lang="en", ocr_version="PP-OCRv4", show_log=False, use_gpu=False)

    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Tampilkan frame dengan OpenCV
        cv2.imshow('Camera Stream', frame)

        # Tangkap gambar ketika tombol spasi ditekan
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # Lakukan OCR pada frame yang diambil
            ocr_result = ocr_from_frame(frame, ocr_model)
            
            # Tampilkan hasil OCR di terminal
            print(ocr_result)

        # Tekan 'q' untuk keluar dari loop
        elif key == ord('q'):
            break

    # Melepaskan sumber daya kamera dan menutup semua jendela
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
