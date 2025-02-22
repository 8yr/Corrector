import os
import cv2
import numpy as np
import fitz
from PIL import Image
import img2pdf
from io import BytesIO

def correct_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    angle = 0
    if lines is not None:
        angles = [(theta * 180 / np.pi) - 90 for rho, theta in lines[:,0]]
        angle = np.median(angles)
    h, w = image.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
def process_file(file_path, output_dir):
    try:
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"corrected_{name}{ext if ext.lower() != '.pdf' else '.pdf'}")
        if ext.lower() in ['.jpg', '.jpeg', '.png']:
            pil_img = Image.open(file_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            corrected_img = correct_rotation(opencv_img)
            cv2.imwrite(output_path, corrected_img)
            print(f"Corrected: {base_name}")
        elif ext.lower() == '.pdf':
            doc = fitz.open(file_path)
            corrected_images = []
            for page in doc:
                mat = fitz.Matrix(300/72, 300/72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                corrected_img = correct_rotation(img_bgr)
                _, buffer = cv2.imencode('.png', cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
                corrected_images.append(buffer.tobytes())
            with open(output_path, "wb") as f:
                f.write(img2pdf.convert([BytesIO(img) for img in corrected_images]))
            print(f"Corrected: {base_name}")
    except Exception as e:
        print(f"Error {base_name}: {str(e)}")
def process_folder(input_path, output_dir="corrected_files"):
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(input_path):
        process_file(input_path, output_dir)
    else:
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
                    file_path = os.path.join(root, file)
                    process_file(file_path, output_dir)
process_folder("inputs", "outputs")
