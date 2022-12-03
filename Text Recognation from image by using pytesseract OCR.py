import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


img = cv2.imread('img2.jpg')

img = cv2.resize(img, (None) , fx=0.5 , fy = 0.5)
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
Adaptive_threshold = cv2.adaptiveThreshold(gray , 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11) 

config = "--psm 3" # page segmentation mode
text = pytesseract.image_to_string(Adaptive_threshold, config = config, lang=None ) # chi_sim

print(text)


cv2.imshow('gray', gray)
cv2.imshow('Adaptive_threshold', Adaptive_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()