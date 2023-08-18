

import cv2
import numpy as np


# def obs2goal_transform(curr_obs, short_goal):

# Load the two images
image1 = cv2.imread('curr_obs_img.jpg', 0)  # color image
image2 = cv2.imread('short_goal_img.jpg', 0)  # color image

image1 = cv2.cvtColor(cv2.cvtColor(image1, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
image2 =  cv2.cvtColor(cv2.cvtColor(image2, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

cv2.imshow("grayscale_img", image1)
cv2.waitKey(0)

# Detect features and compute descriptors
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)


# Match features between the two images
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)


# Apply ratio test to select good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract matching keypoints' coordinates
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Estimate transformation matrix using RANSAC
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Apply transformation matrix to one of the images
transformed_image = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))
# Display the transformed image
cv2.imshow("Transformed Image", transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
