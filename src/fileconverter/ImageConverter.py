import fitz
import os
import cv2
import numpy as np
from PIL import Image 
from ..exception.NotImageError import NotImageError

class ImageConverter:
    DPI = 300 # 화질
    BASE_DPI = 96 # pdf 기본 화질 

    # 파일 형식 확인 
    @staticmethod
    def get_file_type(file_path):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == '.pdf':
            return 'pdf'
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            return 'image'
        else:
            raise NotImageError("변환할 수 없는 파일 형식입니다.")
            return 'other'

    # 입력값을 img_list로 변환
    @staticmethod
    def file_path_to_imglist(file_path):
        file_type = ImageConverter.get_file_type(file_path)
        img_list = []

        if file_type == "pdf":
            img_list = ImageConverter.convert_pdf_to_png(file_path)
        else:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            img_list.append(img)

        return img_list


    # pdf를 png로 변환
    @staticmethod
    def convert_pdf_to_png(file_path):
        scale = ImageConverter.DPI / ImageConverter.BASE_DPI
        matrix = fitz.Matrix(scale, scale)

        document = fitz.open(file_path)
        img_list = []

        for page in document:
            pixmap = page.get_pixmap(matrix=matrix)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            image_np = np.array(image)
            
            if image_np.ndim == 3 and image_np.shape[2] == 3:
                image_np = image_np[:, :, ::-1]
            if image_np.dtype != np.uint8:
                image_np = image_np.astype(np.uint8) 
            image_np = np.ascontiguousarray(image_np) 
            print("type:", type(image_np))          # <class 'numpy.ndarray'> 여야 함
            print("dtype:", image_np.dtype)         # uint8 여야 함
            print("shape:", image_np.shape)         # (H, W, 3) 여야 함
            print("ndim:", image_np.ndim)   
            img_list.append(image_np)

        document.close()
        return img_list

