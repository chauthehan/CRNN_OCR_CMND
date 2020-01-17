#from pyimagesearch.io import HDF5DatasetWriter
import numpy as np 
import argparse
from imutils import paths
import cv2
import os
import imutils
import random

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

def encode_utf8_string(text, length, dic, null_char_id):
    char_ids_padded = [null_char_id]*length
    char_ids_unpadded = [null_char_id]*len(text)
    for i in range(len(text)):
        hash_id = dic[text[i]]
        char_ids_padded[i] = hash_id
        char_ids_unpadded[i] = hash_id
    return char_ids_padded, char_ids_unpadded

width = 300
height = 32
k1 = 300/32

imagePaths = list(paths.list_images('/home/hansama/Documents/crnn/test/'))
print(imagePaths)
random.shuffle(imagePaths)

for i, imagePath in enumerate(imagePaths):
	image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
	k2 = image.shape[1]/image.shape[0]
	if k2 > k1:		
		resized = imutils.resize(image, width = 300)
		zeros = np.zeros((32 - resized.shape[0], 300))
		results = np.concatenate((resized, zeros), axis=0)
	else:
		resized = imutils.resize(image, height = 32)
		zeros = np.zeros((32, 300 - resized.shape[1]))
		results = np.concatenate((resized, zeros), axis=1)
	results = imutils.rotate_bound(results, 90)

	cv2.imwrite(str(i)+'.jpg', results)
	#get the dictionary
	# dic = {}
	# dic[" "] = 0
	# with open('dic.txt', encoding="utf-8") as dict_file:
	#     for i, line in enumerate(dict_file):
	#         if i == 0:
	#             continue

	#         (key, value) = line.strip().split('\t')
	#         dic[value] = int(key)
	# dict_file.close()
 #    #convert label     
	# label = imagePath.split('.')[0] + '.txt'

	# with open(label, 'r') as f:
	# 	for line in f:
	# 		char_ids_padded, char_ids_unpadded = encode_utf8_string(
 #                            text=line,
 #                            dic=dic,
 #                            length=37,
 #                            null_char_id=214)
	# f.close()
	# print(results.shape)
	# print("!!!!!!!!!",char_ids_padded)



