import os
import shutil
from pathlib import Path
from PIL import Image
from coco_dataset_working import classes_indexes_to_names


def rename_images(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    images = []

    # Собираем все файлы с допустимыми расширениями
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            images.append(filename)

    i = 1
    for filename in images:
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"img{i}{'.jpg'}"  # Все переименовываются в .jpg
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        # Проверяем, существует ли файл с таким же именем, как новое имя
        if os.path.exists(new_path):
            i += 1
            print(f"Предупреждение: Файл {new_name} уже существует.  Пропускаем {filename}.")
        else:
            try:
                # Открываем изображение с помощью Pillow, чтобы проверить, что это допустимый файл изображения
                img = Image.open(old_path)
                img.close()  # Закрываем изображение сразу после открытия

                os.rename(old_path, new_path)
                print(f"Переименован {filename} в {new_name}")
                i += 1
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")


def fill_the_dataset(dataset_name, dir_name_images, dir_name_textes, dataset_train_percent = 0.8):
    # Распределение файлов в train и valid  составляющую датасета
    # train - 80% ; valid - 20%

    train_path_images = Path(dataset_name)/"train"/"images"
    train_path_labels = Path(dataset_name)/"train"/"labels"
    valid_path_images = Path(dataset_name)/"valid"/"images"
    valid_path_labels = Path(dataset_name)/"valid"/"labels"

    image_extensions = ('.jpg', '.jpeg', '.png')
    all_imag = [i for i in os.listdir(dir_name_images) if i.lower().endswith(image_extensions)]

    count_of_files = len(all_imag)

    for cur_image_raw in all_imag[:int(count_of_files *dataset_train_percent)]:
        cur_text = cut_the_extension(cur_image_raw) + '.txt'

        cur_image = Path(dir_name_images)/cur_image_raw
        cur_text = Path(dir_name_textes)/cur_text

        shutil.move(cur_image, train_path_images)
        shutil.move(cur_text, train_path_labels)

    for cur_image_raw in all_imag[int(count_of_files * dataset_train_percent):]:
        cur_text = cut_the_extension(cur_image_raw) + '.txt'

        cur_image = Path(dir_name_images)/cur_image_raw
        cur_text = Path(dir_name_textes)/cur_text

        shutil.move(cur_image, valid_path_images)
        shutil.move(cur_text, valid_path_labels)


def create_empty_dataset(dataset_name):
    # Создание структуры папок, удобной для обучения модели
    os.makedirs(os.path.join(dataset_name, 'train', 'labels'))
    os.makedirs(os.path.join(dataset_name, 'train', 'images'))
    os.makedirs(os.path.join(dataset_name, 'valid', 'labels'))
    os.makedirs(os.path.join(dataset_name, 'valid', 'images'))


def create_yaml(data_folder_path: str, class_ides: list[int], class_names):
    yaml_path = Path(data_folder_path)/'data.yaml'
    with open(yaml_path, 'w') as yaml:
        yaml.write('train: ../train/images\n')
        yaml.write('val: ../val/images\n')
        yaml.write(f'nc: {len(class_ides)}\n')
        yaml.write(f'names: [{', '.join([f"'{class_names[i]}'" for i in class_ides])}]\n')


def replace_first_number_in_files(folder_path: str, classes_indexes: list[int]):
    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):  # Обрабатываем только .txt файлы
            filepath = os.path.join(folder_path, filename)
            process_file(filepath, classes_indexes)


def process_file(filepath: str, classes_indexes: list[int]):
    print(filepath)
    key_dict = form_key_dict(classes_indexes)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    print(len(lines))
    for line in lines:

        parts = line.split()  # Разбиваем строку на части по пробелам
        print(len(parts))
        if parts:  # Проверяем, что строка не пустая
            try:
                first_number = int(parts[0])  # Пытаемся преобразовать первый элемент в число
                if first_number in key_dict:
                    parts[0] = str(key_dict[first_number])  # Заменяем, если есть в словаре
                else:
                    print("not in dict!!!")
                new_line = ' '.join(parts)  # Собираем строку обратно
                new_lines.append(new_line)
            except ValueError:
                # Если первый элемент не число, просто добавляем строку как есть
                new_lines.append(line)
        else:
            new_lines.append(line)
        print(len(new_lines))

    with open(filepath, 'w') as f:
        f.writelines([f'{i}\n' for i in new_lines])


def form_key_dict(classes_indexes: str):
    ans = {}
    for i in range(len(classes_indexes)):
        ans[classes_indexes[i]] = i
    return ans

def cut_the_extension(filepath: str):
    return filepath.split('.')[-2]
