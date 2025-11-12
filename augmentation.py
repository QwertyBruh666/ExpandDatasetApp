import shutil
import cv2
import numpy as np
from PIL import ImageOps, ImageEnhance, Image, ImageFile
import os
import random
from ultralytics.data.augment import RandomHSV


def do_base_augm(augmentation: str, dir_name_images, dir_name_textes):
    """Геометрическая аугментация изображений и их разметки."""
    image_files = [f for f in os.listdir(dir_name_images)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(dir_name_images, image_file)
        base_name, ext = os.path.splitext(os.path.basename(image_file))
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"⚠️ Не удалось открыть {image_path}: {e}")
            continue

        text_info = []
        if dir_name_textes:
            text_path = os.path.join(dir_name_textes, base_name + ".txt")
            if os.path.exists(text_path):
                with open(text_path) as f:
                    text_info = [line.split() for line in f.readlines()]

        new_info = []
        if augmentation == 'mirror':
            new_image = ImageOps.mirror(image)
            for line in text_info:
                new_line = line.copy()
                for i in range(1, len(line), 2):
                    new_line[i] = str(1 - float(line[i]))
                new_info.append(' '.join(new_line))

        elif augmentation == 'flip':
            new_image = ImageOps.flip(image)
            for line in text_info:
                new_line = line.copy()
                for i in range(2, len(line), 2):
                    new_line[i] = str(1 - float(line[i]))
                new_info.append(' '.join(new_line))
        else:
            continue

        # Сохранение
        new_image_name = f"{base_name}_{augmentation}{ext}"
        new_image_path = os.path.join(dir_name_images, new_image_name)
        new_image.save(new_image_path)

        if dir_name_textes and new_info:
            new_text_name = f"{base_name}_{augmentation}.txt"
            new_text_path = os.path.join(dir_name_textes, new_text_name)
            with open(new_text_path, 'w') as f:
                f.write('\n'.join(new_info))


def do_hard_augm(augmentation: str, dir_name_images, dir_name_textes):
    """Графическая аугментация (яркость, контраст, резкость)."""
    image_files = [f for f in os.listdir(dir_name_images)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(dir_name_images, image_file)
        base_name, ext = os.path.splitext(os.path.basename(image_file))
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"⚠️ Не удалось открыть {image_path}: {e}")
            continue

        text_info = ""
        if dir_name_textes:
            text_path = os.path.join(dir_name_textes, base_name + ".txt")
            if os.path.exists(text_path):
                with open(text_path) as f:
                    text_info = f.read()

        if augmentation == 'sharpness':
            factors = [0.5, 5]
            enhancer = ImageEnhance.Sharpness(image)
        elif augmentation == 'contrast':
            factors = [0.4, 3]
            enhancer = ImageEnhance.Contrast(image)
        else:
            continue

        for i, factor in enumerate(factors):
            new_image = enhancer.enhance(factor)
            new_image_name = f"{base_name}_{augmentation}{i}{ext}"
            new_image_path = os.path.join(dir_name_images, new_image_name)
            new_image.save(new_image_path)

            if dir_name_textes and text_info:
                new_text_name = f"{base_name}_{augmentation}{i}.txt"
                new_text_path = os.path.join(dir_name_textes, new_text_name)
                with open(new_text_path, 'w') as f:
                    f.write(text_info)


def do_hsv_augm(dir_name_images, dir_name_textes):
    """HSV-аугментация через ручную функцию."""
    factors = [
        (0.7, 1.0, 0.5),
        (0.9, 0.2, 0.3),
        (0.3, 0.7, 0.6),
        (0.9, 0.5, 1.0),
        (0.5, 0.0, 0.8)
    ]

    image_files = [f for f in os.listdir(dir_name_images)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(dir_name_images, image_file)
        base_name, ext = os.path.splitext(os.path.basename(image_file))
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Не удалось прочитать {image_path}")
            continue

        text_info = ""
        if dir_name_textes:
            text_path = os.path.join(dir_name_textes, base_name + ".txt")
            if os.path.exists(text_path):
                with open(text_path) as f:
                    text_info = f.read()

        for i, factor in enumerate(factors):
            new_image_name = f"{base_name}_hsv{i}{ext}"
            new_image_path = os.path.join(dir_name_images, new_image_name)
            augmented_image = augment_hsv(image.copy(), *factor)
            cv2.imwrite(new_image_path, augmented_image)

            if dir_name_textes and text_info:
                new_text_name = f"{base_name}_hsv{i}.txt"
                new_text_path = os.path.join(dir_name_textes, new_text_name)
                with open(new_text_path, 'w') as f:
                    f.write(text_info)


def do_yolohsv_augm(dir_name_images, dir_name_textes):
    """HSV-аугментация через YOLO RandomHSV."""
    factors = [
        (0.9, 0.2, 0.3),
        (0.3, 0.7, 0.6),
        (0.7, 0.4, 0.8),
        (0.5, 0.0, 1.0),
        (0.5, 1.0, 0.0),
    ]

    image_files = [f for f in os.listdir(dir_name_images)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(dir_name_images, image_file)
        base_name, ext = os.path.splitext(os.path.basename(image_file))
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Не удалось прочитать {image_path}")
            continue

        text_info = ""
        if dir_name_textes:
            text_path = os.path.join(dir_name_textes, base_name + ".txt")
            if os.path.exists(text_path):
                with open(text_path) as f:
                    text_info = f.read()

        for i, factor in enumerate(factors):
            hsv_aug = RandomHSV(*factor)
            labels = {'img': image.copy()}
            hsv_aug(labels)
            augmented = labels['img']

            new_image_name = f"{base_name}_yolohsv{i}{ext}"
            new_image_path = os.path.join(dir_name_images, new_image_name)
            cv2.imwrite(new_image_path, augmented)

            if dir_name_textes and text_info:
                new_text_name = f"{base_name}_yolohsv{i}.txt"
                new_text_path = os.path.join(dir_name_textes, new_text_name)
                with open(new_text_path, 'w') as f:
                    f.write(text_info)


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """Применяет HSV-аугментацию к изображению (вручную)."""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue),
                        cv2.LUT(sat, lut_sat),
                        cv2.LUT(val, lut_val)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
    return img


def augmentate_it(dir_name_images: str, dir_name_textes=None):
    """Запуск всех типов аугментаций."""
    list_of_base_augm = ['flip', 'mirror']
    for aug in list_of_base_augm:
        do_base_augm(aug, dir_name_images, dir_name_textes)

    list_of_hard_augm = ['contrast']
    for aug in list_of_hard_augm:
        do_hard_augm(aug, dir_name_images, dir_name_textes)

    do_yolohsv_augm(dir_name_images, dir_name_textes)
    # do_hsv_augm(dir_name_images, dir_name_textes)