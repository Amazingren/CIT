import os
from inception_score import get_inception_score
from skimage.io import imread

def test(generated_IMG_dir):
    print(generated_images_dir)
    print ("Loading image Pairs...")

    generated_images = []
    for img_nameG in os.listdir(generated_IMG_dir):
        imgG = imread(os.path.join(generated_IMG_dir, img_nameG))
        generated_images.append(imgG)

    print("#######IS########")
    print ("Compute inception score...")
    inception_score = get_inception_score(generated_images)
    print ("Inception score %s" % inception_score[0])



if __name__ == "__main__":
    generated_images_dir = '/home/data/try-on/GMM/0309_03/result/TOM/test/try-on'
    #generated_images_dir = '/home/data/try-on/GMM/0309_03/result_ours1/TOM/test/try-on'
    test(generated_images_dir)
