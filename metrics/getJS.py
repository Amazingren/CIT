import numpy as np
import os
from skimage.io import imread, imsave
# from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import jaccard_score


def iou(img1, img2):
	lable_area = 0
	res_area = 0
	intersection_area = 0

	rows, cols = img1.shape[:2]

	for row in range(rows):
		for col in range(cols):
			if img1[row][col] == 255 and img2[row][col] == 255:
				intersection_area += 1
				lable_area +=  1
				res_area += 1
			elif img1[row][col] == 255  and img2[row][col] != 255:
				lable_area += 1
			elif img1 [row][col] != 255 and img2[row][col] == 255:
				res_area += 1
 			
	
	combine_area = lable_area + res_area - intersection_area

	iou = intersection_area / combine_area
	return iou 

def iou1(img1, img2):
	# 
	# for row in range(256):
	# 	for col in range(192):
	# 		if img1[row][col] == 255:
	# 			img1[row][col] = 1


	for row in range(256):
		for col in range(192):
			if img2[row][col] == 127:
				img2[row][col] = 0
			# elif img2[row][col] == 255:
			# 	img2[row][col] = 1


	# labels = [0, 1]
	# jaccards = []
	# for label in labels:
	# 	jaccard = jaccard_score(img2.flatten(), img1.flatten(), pos_label=label, average='micro')
	# 	jaccards.append(jaccard)
	# avg_iou = np.mean(jaccards)

	jaccard = jaccard_score(img2.flatten(), img1.flatten(), average='micro')
	return jaccard


def iou_score(warpedMask_imgs, mask_onPerson_imgs):
	iou_score_list = []
	for warpedMask_img, mask_onPerson_img in zip(warpedMask_imgs, mask_onPerson_imgs):
		# print(type(warpedMask_img))
		# print(type(mask_onPerson_img))
		iou_score = iou1(warpedMask_img, mask_onPerson_img)
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

	print("######IOU######")
	Final_iou_score = iou_score(warpedMask_imgs, mask_onPerson_imgs)
	print("IOU: %s " % Final_iou_score)


if __name__ == "__main__":
	#warpedMask_dir = '/home/data/try-on/GMM/0309_03/result/GMM/test/warp-mask'
	warpedMask_dir = '/home/data/try-on/cp-vton/result/gmm_final.pth/test/warp-mask'
	#mask_onPerson_dir = '/home/data/try-on/GMM/0309_03/result/GMM/test/pcm'
	mask_onPerson_dir = '/home/data/try-on/GMM/0309_03/result_ours2/GMM_ssim/test/pcm'
	

	# # print("###########")


	# img1_path = '/home/data/try-on/cp-vton-plus/result/GMM/test/warp-mask/000028_0.jpg'
	# img2_path = '/home/data/try-on/cp-vton-plus/result/GMM/test/pcm/000028_0.jpg'
	# img1 = imread(img1_path)
	# img2 = imread(img2_path)
	# for row in range(256):
	# 	for col in range(192):
	# 		if img1[row][col] == 255:
	# 			img1[row][col] = 1


	# for row in range(256):
	# 	for col in range(192):
	# 		if img2[row][col] == 127:
	# 			img2[row][col] = 0
	# 		elif img2[row][col] == 255:
	# 			img2[row][col] = 1


	# print(iou(imread(img1_path), imread(img2_path)))
	# jaccard = jaccard_score(img2.flatten(), img1.flatten(), average='micro')
	# print("avg_iou:", jaccard)


	# labels = [0, 1]
	# jaccards = []


	# for label in labels:
	# 	jaccard = jaccard_score(img2.flatten(), img1.flatten(), pos_label=label, average='micro')
		
		
	# 	jaccards.append(jaccard)

	# avg_iou = np.mean(jaccards)

	# print("avg_iou:", avg_iou)



	# masks = np.reshape(img2 > 127, (-1, 1)).astype(np.float32)



	# print(type(img1))
	# print(type(img2))

	# jac = jaccard_score(img1, img2, average='samples')

	# print(iou1(imread(img1_path), imread(img2_path)))

	# print("Jac", jac)
	test(warpedMask_dir, mask_onPerson_dir)


