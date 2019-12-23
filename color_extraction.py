from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import cv2
import numpy as np

def get_dominant_color(image, k, image_processing_size = None):
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)
    
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)

    label_counts = Counter(labels)
    most_common = label_counts.most_common(3)
    dominant_colors = []
    for element in most_common:
        dominant_colors.append(clt.cluster_centers_[element[0]])
    return dominant_colors

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow("capture",frame)
cv2.waitKey(0)
#while(True):
#    ret, frame = cap.read()
#    cv2.imshow('frame',frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
cap.release()
cv2.destroyAllWindows()

image_processing_size=(500,500)
image = frame
#image = cv2.imread('image5.jpeg')
image = cv2.resize(image, image_processing_size, interpolation = cv2.INTER_AREA)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
dom_colors = get_dominant_color(hsv_image, k=5)
dom_color_bgr = []
for dom_color in dom_colors:
    dom_color_hsv = np.full((200,200,3), dom_color, dtype='uint8')
    dom_color_bgr.append(cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2BGR))

output_image = np.hstack(img for img in dom_color_bgr)
cv2.imshow('Image Dominant Color', output_image)
cv2.waitKey(0)