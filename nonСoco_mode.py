import time
from pathlib import Path
import numpy as np
import torch
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
import os
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

from file_working import create_empty_dataset, fill_the_dataset, create_yaml
from augmentation import augmentate_it

def create_annotated_images(images_folder, detections_folder, output_folder, classes = None, class_colors=None):
    """
    Создает папку с изображениями, на которых нанесены результаты детекции.

    :param images_folder: Путь к папке с исходными изображениями.
    :param detections_folder: Путь к папке с файлами результатов детекции.
    :param output_folder: Путь к папке, куда сохранять размеченные изображения.
    :param class_colors: Опционально, словарь для назначения цветов классам, например {0: (0,255,0), 1: (0,0,255)}
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Получаем список изображений
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.HEIC'))]

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        detection_file = os.path.splitext(image_file)[0] + '.txt'
        detection_path = os.path.join(detections_folder, detection_file)

        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение {image_path}")
            continue

        # Проверяем наличие файла детекции
        if not os.path.exists(detection_path):
            print(f"Файл детекции не найден: {detection_path}")
            # Можно пропустить или оставить изображение без изменений
            # Для этого закомментируйте следующую строку
            # pass
            # И продолжить
            cv2.imwrite(os.path.join(output_folder, image_file), image)
            continue

        # Читаем файл детекций
        with open(detection_path, 'r') as f:
            lines = f.readlines()

        height, width = image.shape[:2]
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x1_rel, y1_rel, x2_rel, y2_rel = parts
            class_id = int(class_id)
            x1_rel, y1_rel, x2_rel, y2_rel = map(float, [x1_rel, y1_rel, x2_rel, y2_rel])

            # Преобразуем относительные координаты в абсолютные
            x1_abs = int(x1_rel * width)
            y1_abs = int(y1_rel * height)
            x2_abs = int(x2_rel * width)
            y2_abs = int(y2_rel * height)

            # Цвет для класса
            color = (0, 255, 0)  # по умолчанию зеленый
            if class_colors and class_id in class_colors:
                color = class_colors[class_id]

            # Рисуем прямоугольник
            cv2.rectangle(image, (x1_abs, y1_abs), (x2_abs, y2_abs), color, 2)
            # Можно добавить подпись класса
            if classes:
                class_id = classes[class_id]
            cv2.putText(image, str(class_id), (x1_abs, y1_abs - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Сохраняем изображение
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)


def create_segmentated_annotated_images(image_dir, text_dir, output_dir):
    os.makedirs(output_dir)
    image_extensions = ('.jpg', '.jpeg', '.png')
    all_images_paths = [
        os.path.join(image_dir, image_name)
        for image_name in os.listdir(image_dir)
        if image_name.lower().endswith(image_extensions)
    ]
    all_textes_paths = [
        os.path.join(text_dir, text_name)
        for text_name in os.listdir(text_dir)
        if text_name.lower().endswith('.txt')
    ]

    for (image_path, text_path) in zip(all_images_paths, all_textes_paths):
        print(image_path, text_path)

        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Шаг 2: Чтение файла с разметкой

        with open(text_path, 'r') as f:
            lines = f.readlines()
        # Набор ярких цветов (BGR формат)
        colors = [
            (255, 0, 0),  # синий
            (0, 255, 0),  # зелёный
            (0, 0, 255),  # красный
            (255, 255, 0),  # жёлтый
            (255, 0, 255),  # пурпурный
            (0, 255, 255),  # голубой
            (128, 0, 128),  # фиолетовый
            (0, 128, 128),  # тёмно-бирюзовый
            (128, 128, 0),  # оливковый
            (0, 0, 128),  # тёмно-синий
        ]

        def get_color(index):
            return colors[index % len(colors)]

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            label = parts[0]
            coords = list(map(float, parts[1:]))

            if len(coords) % 2 != 0:
                print(f"Ошибка: в строке с {label} нечётное количество координат")
                continue

            points = []
            for i in range(0, len(coords), 2):
                x_rel = coords[i]
                y_rel = coords[i + 1]
                x_abs = int(x_rel * width)
                y_abs = int(y_rel * height)
                points.append([x_abs, y_abs])

            points = np.array(points, dtype=np.int32)

            color = get_color(idx)

            # Нарисовать контур
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

            # Заливка с прозрачностью
            overlay = image.copy()
            cv2.fillPoly(overlay, [points], color=color)
            alpha = 0.3
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # Надпись с меткой
            cv2.putText(image, label, (points[0][0], points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2)

        new_image_name = Path(output_dir)/Path(image_path).name
        cv2.imwrite(str(new_image_name), image)


def zero_shot_folder_detection(image_dir: str, classes: list[str], output_dir=None, min_confidence: float = 0.03,
                               model_id: str = "yolo_world/l"):
    output_dir = output_dir or f'{image_dir}_texts'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = YOLOWorld(model_id=model_id)
    model.set_classes(classes)

    # Список расширений изображений
    image_extensions = ('.jpg', '.jpeg', '.png', '.heic')

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(image_dir, filename)
            zero_shot_image_detection(file_path, model, output_dir, min_confidence)


def zero_shot_image_detection(image_path: str, model, output_dir, min_confidence: float = 0.03):
    image = cv2.imread(image_path)
    results = model.infer(image, confidence=min_confidence)
    detections = sv.Detections.from_inference(results)
    hints = detections.xyxy.tolist()
    y, x, _ = image.shape
    for i in range(len(hints)):
        hints[i][0] /= x
        hints[i][2] /= x
        hints[i][1] /= y
        hints[i][3] /= y
    info = [
        f"{classid} {' '.join([str(i) for i in hint])}\n"
        for classid, hint
        in zip(detections.class_id, hints)
    ]
    print(image_path)
    image_path = image_path.split('/')[-1]
    print(image_path)
    # text_filename = f"{output_dir}/{image_path.replace(os.path.splitext(image_path)[1], ".txt")}"
    text_filename = Path(output_dir)/f"{image_path.split('.')[-2]}.txt"

    with open(text_filename, 'w') as f:
        f.writelines(info)


def sam_folder_segmentation(image_dir: str, texts_dir: str, sam_checkpoint="sam_vit_l_0b3195.pth", model_type="vit_l"):
    sys.path.append("..")
    device = "cpu"
    if torch.cuda.is_available():
        device = 'cuda'
        print('cuda is active')

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    image_extensions = ('.jpg', '.jpeg', '.png', '.HEIC')
    all_images_paths = [
        os.path.join(image_dir, filename)
        for filename in os.listdir(image_dir)
        if filename.lower().endswith(image_extensions)
    ]
    for image_path in all_images_paths:
        sam_image_segmentation(predictor, image_path, texts_dir)


def sam_image_segmentation(predictor, image_path: str, texts_dir: str):
    image_base_path = image_path.split('.')
    image_base_path[-1] = 'txt'
    image_base_path = ('.'.join(image_base_path)).split('/')[-1]
    text_filename = Path(texts_dir)/image_base_path

    # text_filename = f"{texts_dir}/{image_path.replace(os.path.splitext(image_path)[1], ".txt")}"

    with open(text_filename, 'r') as f:
        lines = f.readlines()
    lines = [(line.rstrip()).split() for line in lines]
    '''
    for line in lines:
        class_id = int(line[0])
        hints = [int(i) for i in line[1:]]
    '''
    bboxes = [(int(line[0]), [float(i) for i in line[1:]]) for line in lines]

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    y, x, _ = image.shape
    information = ''
    for bbox in bboxes:
        class_number, xyxy = bbox
        xyxy[0] *= x
        xyxy[1] *= y
        xyxy[2] *= x
        xyxy[3] *= y
        input_box = np.array(xyxy)

        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )

        h, w = masks.shape[1], masks.shape[2]

        # Объединим все маски в одну с помощью логического ИЛИ
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            combined_mask = cv2.bitwise_or(combined_mask, m.astype(np.uint8))

        # Находим контуры объединённой маски
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            print("Контуры не найдены")
            continue

        # Выбираем контур с максимальным числом точек (главный контур)
        contour = max(contours, key=len)
        contour_coords = contour[:, 0, :]  # shape: (N, 2)

        # Преобразование в относительные координаты (x/w, y/h)
        contour_coords_rel = contour_coords.astype(np.float32)
        contour_coords_rel[:, 0] /= w  # нормализация по ширине
        contour_coords_rel[:, 1] /= h  # нормализация по высоте

        # Преобразуем в "плоский" список [x1, y1, x2, y2, ...]
        contour_coords_rel_list = contour_coords_rel.flatten().tolist()

        information += f"{class_number} {' '.join([str(i) for i in contour_coords_rel_list])}\n"

    with open(text_filename, 'w') as f:
        f.writelines(information)


def non_coco_mode(images_folder: str, classes: list[str], enable_sementation: bool, enable_augmentation: bool, enable_dataset:bool,  dataset_train_percent:float, dataset_name=None):
    # Здесь и далее все строчки со временем можешь удалить или использовать, если планируется вывод пользователю затраченного времени
    t_start = time.time()
    dataset_name = dataset_name or 'dataset'
    texts_folder = f'{images_folder}_texts'
    print("Размечаю...")
    t11 = time.time()
    zero_shot_folder_detection(images_folder, classes, texts_folder, min_confidence=0.002)
    t12 = time.time()
    print(f"Разметка завершена! Времени затрачено - {t12-t11}, времени прошло - {t12 - t_start}")

    if enable_sementation:
        print("Сегментирую...")
        t21 = time.time()
        sam_folder_segmentation(images_folder, texts_folder)
        t22 = time.time()
        print(f"Сегментация завершена! Времени затрачено - {t22-t21}, времени прошло - {t22-t_start}")

    if enable_augmentation:
        print("Аугментирую...")
        t31 = time.time()
        augmentate_it(dir_name_images=images_folder, dir_name_textes=texts_folder)
        t32 = time.time()
        print(f"Аугментация завершена! Времени затрачено - {t32-t31}, времени прошло - {t32-t_start}")
    if enable_dataset:
        print("Собираю датасет...")
        create_empty_dataset(dataset_name)
        fill_the_dataset(dataset_name, images_folder, texts_folder, dataset_train_percent)
        create_yaml(dataset_name, list(range(len(classes))), classes)
    print("Готово!")
    print(f"Общее затраченное время - {time.time() - t_start}")


if __name__ == "__main__":
    images_folder = 'data'
    classes = ['eye', 'nose of human']
    enable_sementation = True
    enable_augmentation = True
    enable_dataset = False
    dataset_name = 'dataset'
    dataset_train_percent = 0.8
    non_coco_mode(images_folder, classes, enable_sementation, enable_augmentation, enable_dataset,  dataset_train_percent, dataset_name)
    if not enable_sementation and not enable_dataset:
        create_annotated_images(images_folder, f"{images_folder}_texts", f"{images_folder}_labeled", classes)
    elif enable_sementation and not enable_dataset:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        create_segmentated_annotated_images(os.path.join(current_dir, images_folder),
                                            os.path.join(current_dir, f"{images_folder}_texts"),
                                            os.path.join(current_dir, f"{images_folder}_labeled_segm"))
