import numpy as np
import os

def readFileNames():
	data_folder = '/mnt/sdb1/intern_data/pix2pix_wbce_pet3d/train'
	num_files = len(os.listdir(data_folder))
	input_imgs = []
	target_imgs = []
	for r in range(num_files//2):
		input_imgs.append(data_folder + '/' + str(r) + '_input.npy')
		target_imgs.append(data_folder + '/' + str(r) + '_target.npy')
	return np.array(input_imgs), np.array(target_imgs)

def batchIterator(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs),dtype = np.uint32)
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        input_files = inputs[excerpt]
        target_files = targets[excerpt]
        input_images = []
        target_images = []
        for i in range(len(input_files)):
            input_images.append(np.load(input_files[i])[np.newaxis,:])
            target_arr = np.load(target_files[i])
            shape = target_arr.shape
            target_arr = target_arr[shape[0]//2-14:shape[0]//2+14, shape[1]//2-22:shape[1]//2+22, shape[2]//2-22:shape[2]//2+22]
            target_images.append(target_arr[np.newaxis,:])

        yield np.array(input_images), np.array(target_images)

def genBatchIterator(inputs, batchsize, shuffle=True):
    # this function is not in use right now
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        input_files = inputs[excerpt]
        input_images = []
        for i in range(len(input_files)):
        	input_images.append(np.load(input_files[i])[np.newaxis,:])
        yield np.array(input_images)