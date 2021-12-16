try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import cv2

def ocr_core():
    """
    This function will handle the core OCR processing of images.
    """
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    cv2.imwrite('ocr.jpg',frame)
    text = pytesseract.image_to_string(Image.open('ocr.jpg'))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

