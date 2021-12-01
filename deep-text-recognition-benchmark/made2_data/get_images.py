import shutil

data_root_path = '../../kor_dataset/finetuning_data/made2/images/'
save_root_path = './images/'

# copy images from dataset directory to current directory
shutil.copytree(data_root_path, save_root_path)

# separate dataset : train, validation, test
obj_list = ['train', 'test', 'validation']
for obj in obj_list:
    with open(f'gt_{obj}.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            file_path = line.split('.jpg')[0]
            file_name = file_path.split('/')[1] + '.jpg'
            res = shutil.move(save_root_path+file_name, f'./{obj}/')