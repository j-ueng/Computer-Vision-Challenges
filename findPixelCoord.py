import cv2


def findCentroid(image):

    img = cv2.imread(image)
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # binarise image
    ret, thresh = cv2.threshold(img2, 220, 255, 0)

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        print(M['m00'])
        if M["m00"] > 100:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(cX, cY)
            cv2.circle(img, (cX, cY), 5, (255, 0, 255), -1)
            cv2.putText(img, '({:d}.{:d})'.format(cX, cY), (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1500, 1500)
            cv2.imshow('image', img)
            cv2.waitKey(0)

    cv2.imwrite(image.split('.')[0] + '_centroids.jpg', img)
