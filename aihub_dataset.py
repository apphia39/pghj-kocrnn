import json
import random
import os
from tqdm import tqdm

data_type = 'htr'  # htr, ocr
labeling_filename = 'handwriting_data_info1.json'

## Check Json File
file = json.load(open(f'./kor_dataset/aihub_data/{data_type}/{labeling_filename}'))

## Separate dataset - train, validation, test
image_files = os.listdir(f'./kor_dataset/aihub_data/{data_type}/images/') 
total = len(image_files)

random.shuffle(image_files)

n_train = int(len(image_files) * 0.7)
n_validation = int(len(image_files) * 0.15)
n_test = int(len(image_files) * 0.15)

print(n_train, n_validation, n_test)

train_files = image_files[:n_train]
validation_files = image_files[n_train:n_train+n_validation]
test_files = image_files[-n_test:]

## Separate image id - train, validation, test
train_img_ids = {}
validation_img_ids = {}
test_img_ids = {}

for image in file['images']: # {filename}: {image id}
  if image['file_name'] in train_files:
    train_img_ids[image['file_name']] = image['id']
  elif image['file_name'] in validation_files:
    validation_img_ids[image['file_name']] = image['id']
  elif image['file_name'] in test_files:
    test_img_ids[image['file_name']] = image['id']

## Annotations - train, validation, test 
train_annotations = {f:[] for f in train_img_ids.keys()} # {image id}: []
validation_annotations = {f:[] for f in validation_img_ids.keys()}
test_annotations = {f:[] for f in test_img_ids.keys()}

train_ids_img = {train_img_ids[id_]:id_ for id_ in train_img_ids}
validation_ids_img = {validation_img_ids[id_]:id_ for id_ in validation_img_ids}
test_ids_img = {test_img_ids[id_]:id_ for id_ in test_img_ids}

for idx, annotation in enumerate(file['annotations']):
  if idx % 5000 == 0:
    print(idx,'/',len(file['annotations']),'processed')
  if annotation['image_id'] in train_ids_img:
    train_annotations[train_ids_img[annotation['image_id']]].append(annotation)
  elif annotation['image_id'] in validation_ids_img:
    validation_annotations[validation_ids_img[annotation['image_id']]].append(annotation)
  elif annotation['image_id'] in test_ids_img:
    test_annotations[test_ids_img[annotation['image_id']]].append(annotation)

## Write json files
with open(f'{data_type}_train_annotation.json', 'w') as file:
  json.dump(train_annotations, file)
with open(f'{data_type}_validation_annotation.json', 'w') as file:
  json.dump(validation_annotations, file)
with open(f'{data_type}_test_annotation.json', 'w') as file:
  json.dump(test_annotations, file)

## Make gt_xxx.txt files
data_root_path = f'./kor_dataset/aihub_data/{data_type}/images/'
save_root_path = f'./deep-text-recognition-benchmark/{data_type}_data/'

obj_list = ['test', 'train', 'validation']
for obj in obj_list:
  total_annotations = json.load(open(f'./{data_type}_{obj}_annotation.json'))
  gt_file = open(f'{save_root_path}gt_{obj}.txt', 'w')
  for file_name in tqdm(total_annotations):
    annotations = total_annotations[file_name]
    for idx, annotation in enumerate(annotations):
      text = annotation['text']
      gt_file.write(f'{obj}/{file_name}\t{text}\')
