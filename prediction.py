import argparse
from keras.models import load_model
from create_model import CRNN
from config import config
from pyimagesearch.io import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
import numpy as np
import cv2
import itertools

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def fastdecode(y_pred, chars):
    results_str = ""
    confidence = 0.0

    for i,one in enumerate(y_pred[0]):

        if one<214 and (i==0 or (one!=y_pred[0][i-1])):
            results_str+= chars[one]

    return results_str
def decode_label(label, chars):
	results_str = ""
	for i, num in enumerate(label[0]):
		if num != 214:
			results_str += chars[num]
	return results_str 

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",
	help = "path to pre-trained model")
ap.add_argument("-i", "--images", 
	help = "path to hdf5 image file ")
args = vars(ap.parse_args())

print('loading model...')
model = CRNN.build(width=32, height=300, depth=1,
		classes=config.NUM_CLASSES, training=0)
model.load_weights(args["model"])

testAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

testGen = HDF5DatasetGenerator(args["images"], config.BATCH_SIZE,
	aug=testAug, preprocessors=[iap], binarize=False,  classes=config.NUM_CLASSES)

dic = {}
dic[0] = ' '
with open('dic.txt', encoding="utf-8") as dict_file:
	for i, line in enumerate(dict_file):
		if i == 0:
			continue
		(key, value) = line.strip().split('\t')
		dic[int(key)] = value
dict_file.close()

acc = 0
total = 0

print('predicting...')
for (i, (inputs, _)) in enumerate(testGen.generator(passes=1)):

	for j, image in enumerate(inputs['input']):

		predict = model.predict([[image]])	
		predict = np.argmax(predict, axis=2)		

		res = fastdecode(predict, dic)

		label = decode_label(inputs['label'][j].reshape([-1, 37]), dic)
		print('label= ', label)
		print('result=', res)
		# img = rotate_bound(image, -90)		
		# cv2.imshow("q", img)
		# cv2.waitKey(0)		
		# cv2.destroyAllWindows()

		if res == label:
			acc = acc + 1
		total = total+1

print('accuracy=', acc)
print('total=', total)
print('percent=', acc/total)
testGen.close()
