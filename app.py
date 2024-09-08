from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List

from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2


app = FastAPI()


def register_and_compare(im1, im2):
	imgs = [im1, im2]

	# identify s/l images based on area
	areas = []
	for im in imgs:
		h, w = im.shape[0], im.shape[1]
		areas.append(h*w)

	max_index = areas.index(max(areas))
	min_index = 0 if max_index == 1 else 1
	im_l = imgs[max_index]
	im_s = imgs[min_index]

	# Initialize SIFT detector
	sift = cv2.SIFT_create()
	# Detect keypoints and descriptors
	kp1, des1 = sift.detectAndCompute(im_l, None)
	kp2, des2 = sift.detectAndCompute(im_s, None)

	# Check if descriptors were found
	if des1 is None or des2 is None:
		return "No descriptors found in one or both images."
	# get matches and apply 
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)
	good_matches = []
	for m,n in matches:
	    if m.distance < 0.7*n.distance:
	        good_matches.append(m)

	# Extract location of good matches
	src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
	# Compute homography
	H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	# Warp img1 to align with img2
	height, width = im_s.shape[0], im_s.shape[1]
	warped_img = cv2.warpPerspective(im_l, H, (width, height))
	si, _ = ssim(warped_img, im_s, full=True)

	if si >= 0.9:
		return f"match"

	else:
		return f"not a match"

@app.post("/compare-images")
async def compare_images(files: List[UploadFile] = File(...)):
	# Ensure exactly two images are uploaded
	if len(files) != 2:
		raise HTTPException(status_code=400, detail="Exactly two images must be uploaded.")

	try:
		# Read the images from the uploaded files
		image1 = await files[0].read()
		image2 = await files[1].read()

		# Convert images to numpy arrays
		nparr1 = np.frombuffer(image1, np.uint8)
		nparr2 = np.frombuffer(image2, np.uint8)

		# Decode the numpy arrays to OpenCV images
		im1 = cv2.imdecode(nparr1, cv2.IMREAD_GRAYSCALE)
		im2 = cv2.imdecode(nparr2, cv2.IMREAD_GRAYSCALE)

		# Check if images were loaded correctly
		if im1 is None or im2 is None:
			raise HTTPException(status_code=400, detail="One or both images could not be read.")

		# Compare the images using SIFT
		result = register_and_compare	(im1, im2)

		return {"result": result}

	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# To run the application, use: uvicorn script_name:app --reload