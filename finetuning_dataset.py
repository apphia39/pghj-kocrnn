import random
import os

data_type = 'made2' # made1, made2

# Data Preprocessing
with open(f'./kor_dataset/finetuning_data/{data_type}/labels.txt', 'r') as f:
  labels = f.readlines()

image_files = os.listdir('./kor_dataset/finetuning_data/{data_type}/images/')
total = len(image_files)

random.shuffle(image_files)

n_train = int(len(image_files) * 0.7)
n_validation = int(len(image_files) * 0.15)
n_test = int(len(image_files) * 0.15)

train_files = image_files[:n_train]
validation_files = image_files[n_train:n_train+n_validation]
test_files = image_files[-n_test:]

save_root_path = f'./deep-text-recognition-benchmark/{data_type}_data/'

gt_test = open(save_root_path+'gt_test.txt', 'w')
gt_validation = open(save_root_path + 'gt_validation.txt', 'w')
gt_train = open(save_root_path + 'gt_train.txt', 'w')

for line in labels:
    file_name, annotation = line.split('.jpg')
    file_name += '.jpg'
    print(file_name, annotation, end='')

    if file_name in train_files:
        gt_train.write("train/{}\t{}".format(file_name, annotation))
    if file_name in test_files:
        gt_test.write("test/{}\t{}".format(file_name, annotation))
    elif file_name in validation_files:
        gt_validation.write("validation/{}\t{}".format(file_name, annotation))

print('gt_file done')
