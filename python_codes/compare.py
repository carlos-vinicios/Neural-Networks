import cv2
import os

path_image = "Bases ISIC/3000_normais/Archive/normais/"
path_roi = "Bases ISIC/3000_normais/Archive_rois/normais_novice_masks/"

images = os.listdir(path_image)
rois = os.listdir(path_roi)

for roi in rois:
    for image in images:
        if roi == image:
            image = cv2.imread(path_image + image)
            roi_image = cv2.imread(path_roi + roi)
            cv2.namedWindow("Image")
            cv2.imshow("Image", cv2.resize(image, (900, 700)))
            cv2.moveWindow("Image", 5, 20)
            cv2.namedWindow("Roi")
            cv2.imshow("Roi", cv2.resize(roi_image, (900, 700)))
            cv2.moveWindow("Image", 900, 20)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 255:
                os.remove(path_roi + roi)
                print("Apagou")
            break