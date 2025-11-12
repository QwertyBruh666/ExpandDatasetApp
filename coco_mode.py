from file_working import rename_images, create_empty_dataset, fill_the_dataset, create_yaml, replace_first_number_in_files
from auto_annotating import coco_annotate_it
from augmentation import augmentate_it
import time
from coco_dataset_working import COCO_dataset_info
'''
input:
* images folder
* classes indexes

output:
* dataset
* * train
* * * images
* * * textes
* * valid
* * * images
* * * textes
* * yaml
'''


def coco_mode(images_folder:str, classes_indexes: list[int]):
    texts_folder = f'{images_folder}_texts'
    # rename_images(images_folder) # Переименование изображений
    coco_annotate_it(input_dir=images_folder, output_dir=texts_folder)  # Создание текстовых файлов
    replace_first_number_in_files(texts_folder, classes_indexes)
    augmentate_it(dir_name_images=images_folder, dir_name_textes=texts_folder)
    '''
    dataset_name = 'dataset'
    create_empty_dataset(dataset_name)
    fill_the_dataset(dataset_name, images_folder, texts_folder)
    create_yaml(dataset_name, classes_indexes, COCO_dataset_info)
    '''

if __name__ == '__main__':
    #t1 = time.time()
    images_folder = 'data'
    classes_indexes = [0, 27]
    coco_mode(images_folder, classes_indexes)
    #print(time.time()-t1)