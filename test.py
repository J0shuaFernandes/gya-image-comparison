from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
	
	#plt.imshow(warped_img)
	#plt.show()

	si, _ = ssim(warped_img, im_s, full=True)

	return si


for i in list(range(7)):
	# read images
	im1 = cv2.imread(f'img/{i}_source.jpg')
	im2 = cv2.imread(f'img/{i}_comp.jpg')

	im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
	im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

	score = register_and_compare(im1, im2)
	score = round(score, 2)
	print(f'{i}:{score}')
	# Create a figure and axis array
	fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
	# Display each image
	axes[0].imshow(im1_rgb)
	axes[0].axis('off') 
	axes[0].set_title('Image 1')  # Optional: Add a title

	axes[1].imshow(im2_rgb)
	axes[1].axis('off')
	axes[1].set_title('Image 2')

	# Display the comparison result in the center between the two images
	fig.text(0.5, 0.5, f'Score: {score}', ha='center', va='center', fontsize=12, transform=plt.gcf().transFigure)
	plt.tight_layout()
	plt.show()
	# Save the plot as a PNG
	fig.savefig(f'img/output_{i}.png')
	plt.close(fig)