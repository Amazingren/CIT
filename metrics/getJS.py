import numpy as np
import os
from skimage.io import imread, imsave
from sklearn.metrics import jaccard_score


def computeJS(img1, img2):
	# Adjust the pixel value 
	for row in range(256):
		for col in range(192):
			if img2[row][col] == 127:
				img2[row][col] = 0

	jaccard = jaccard_score(img2.flatten(), img1.flatten(), average='micro')
	return jaccard


def JS_score(warpedMask_imgs, mask_onPerson_imgs):
	iou_score_list = []
	for warpedMask_img, mask_onPerson_img in zip(warpedMask_imgs, mask_onPerson_imgs):
		iou_score = computeJS(warpedMask_img, mask_onPerson_img)
		print(iou_score)
		iou_score_list.append(iou_score)

	return np.mean(iou_score_list)


def test(warpedMask_dir, mask_onPerson_dir):
	print("Loading Images...")
	warpedMask_imgs = []
	for img_nameWM in os.listdir(warpedMask_dir):
		imgWM = imread(os.path.join(warpedMask_dir, img_nameWM))
		warpedMask_imgs.append(imgWM)

	mask_onPerson_imgs = []
	for img_nameOP in os.listdir(mask_onPerson_dir):
		imgOP = imread(os.path.join(mask_onPerson_dir, img_nameOP))
		mask_onPerson_imgs.append(imgOP)

	print("######JS######")
	Final_JS_score = JS_score(warpedMask_imgs, mask_onPerson_imgs)
	print("JS: %s " % Final_JS_score)


if __name__ == "__main__":
	warpedMask_dir = '/path to your CIT foder/result/GMM/test/warp-mask'
	mask_onPerson_dir = '/path to your CIT foder/result/GMM/test/pcm'
	
	test(warpedMask_dir, mask_onPerson_dir)


