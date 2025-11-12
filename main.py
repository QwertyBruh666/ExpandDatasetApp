import sys
import cv2
import time
from pathlib import Path
import numpy as np
import zipfile
import tempfile
import os
import math
import random
import shutil
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QCoreApplication
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QSlider, QLabel, QFileDialog, QGraphicsView, QDialog, QCheckBox, QLineEdit,
                             QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QInputDialog, QFileDialog, QMessageBox, QSizePolicy, QColorDialog)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen
from PyQt5.QtCore import Qt, QRectF

import torch
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
from file_working import create_empty_dataset, fill_the_dataset, create_yaml
from augmentation import augmentate_it

from nonСoco_mode import zero_shot_folder_detection, zero_shot_image_detection, sam_folder_segmentation, sam_image_segmentation

#беды с именованием

#окошко для показа автосегментированных изображений
class SegmentedViewerDialog(QDialog):
    def __init__(self, image_paths, label_paths, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.current_index = 0

        layout = QVBoxLayout(self)
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.show_func()

    def show_func(self):
        print(self.current_index)
        image_path = self.image_paths[self.current_index]
        annotation_path = self.label_paths[self.current_index]
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Шаг 2: Чтение файла с разметкой


        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        # Набор ярких цветов (BGR формат)
        colors = [
            (255, 0, 0),     # синий
            (0, 255, 0),     # зелёный
            (0, 0, 255),     # красный
            (255, 255, 0),   # жёлтый
            (255, 0, 255),   # пурпурный
            (0, 255, 255),   # голубой
            (128, 0, 128),   # фиолетовый
            (0, 128, 128),   # тёмно-бирюзовый
            (128, 128, 0),   # оливковый
            (0, 0, 128),     # тёмно-синий
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
                y_rel = coords[i+1]
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        h, w, ch = image.shape
        qimg = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio
        ))


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Down:
            if self.current_index < len(self.image_paths) - 1:
                self.current_index += 1
                self.show_func()
        elif event.key() == Qt.Key_Up:
            if self.current_index > 0:
                self.current_index -= 1
                self.show_func()
        else:
            super().keyPressEvent(event)

#окошко для показа фоток ручной разметки
class ResultViewerDialog(QDialog):
    def __init__(self, image_paths, label_paths, class_names=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Результаты (изображения + разметка)")
        self.setFixedSize(800, 600)

        self.image_paths = sorted(image_paths)
        self.label_paths = sorted(label_paths)
        self.current_index = 0
        self.class_names = class_names or {}

        self.palette = [
            (220, 20, 60), (0, 128, 0), (30, 144, 255), (255, 165, 0),
            (138, 43, 226), (0, 206, 209), (255, 20, 147), (139, 69, 19),
            (255, 255, 0), (0, 191, 255), (127, 255, 0), (255, 105, 180),
            (70, 130, 180), (244, 164, 96), (0, 255, 127), (199, 21, 133),
            (112, 128, 144), (255, 69, 0), (46, 139, 87), (123, 104, 238),
        ]

        layout = QVBoxLayout(self)
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.load_image()

    def get_color(self, cls_id):
        return self.palette[cls_id % len(self.palette)]

    def load_boxes(self, label_file):
        """Читает YOLO-текстовик"""
        boxes = []
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        boxes.append((int(cls), x, y, w, h))
        return boxes

    def draw_boxes(self, img, boxes):
        """Рисует bbox"""
        H, W, _ = img.shape
        for cls, x, y, w, h in boxes:
            x1 = int((x - w / 2) * W)
            y1 = int((y - h / 2) * H)
            x2 = int((x + w / 2) * W)
            y2 = int((y + h / 2) * H)

            color = self.get_color(cls)
            label = self.class_names.get(cls, str(cls))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    def load_image(self):
        if not self.image_paths:
            return

        img_path = self.image_paths[self.current_index]

        base = os.path.splitext(os.path.basename(img_path))[0]
        label_file = None
        for lf in self.label_paths:
            if os.path.splitext(os.path.basename(lf))[0] == base:
                label_file = lf
                break

        img = cv2.imread(img_path)
        if img is None:
            return

        boxes = self.load_boxes(label_file) if label_file else []
        img = self.draw_boxes(img, boxes)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio
        ))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Down:
            if self.current_index < len(self.image_paths) - 1:
                self.current_index += 1
                self.load_image()
        elif event.key() == Qt.Key_Up:
            if self.current_index > 0:
                self.current_index -= 1
                self.load_image()
        else:
            super().keyPressEvent(event)


class WorkerThread(QThread):
    progress = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if self.fn:
            self.fn(self.progress, *self.args, **self.kwargs)

class Overlay(QDialog):
    def __init__(self, message="Загрузка..."):
        super().__init__()
        layout = QVBoxLayout(self)

        self.current_image_path = 0

        self.label = QLabel(message, self)
        self.label.setStyleSheet("color: black; font-size: 18px;")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label, alignment=Qt.AlignCenter)

    def set_message(self, text):
        self.label.setText(text)

class MyDialog(QDialog):
    def __init__(self):
        self.classname_list = []

        super().__init__()
        self.overlay = Overlay("Начинаю работать...")
        self.setWindowTitle("Auto Dataset settings")
        layout = QVBoxLayout()

        self.enable_augm_checkbox = QCheckBox("Set augmentation")
        self.enable_segm_checkbox = QCheckBox("Set segmentation")

        self.add_button = QPushButton("+ append classname")
        self.add_button.clicked.connect(self.add_class_field)

        self.get_dataset_btn = QPushButton("get dataset")
        self.get_dataset_btn.clicked.connect(self.get_dataset)

        # контейнер для полей
        self.fields_layout = QVBoxLayout()

        # добавляем первый input сразу
        self.add_class_field()

        # собираем интерфейс
        layout.addLayout(self.fields_layout)
        layout.addWidget(self.add_button)

        layout.addWidget(self.get_dataset_btn)
        layout.addWidget(self.enable_augm_checkbox)
        layout.addWidget(self.enable_segm_checkbox)

        self.setLayout(layout)

    def non_coco_mode(self, progress_signal,images_folder: str, classes: list[str], enable_sementation: bool, enable_augmentation: bool, enable_dataset:bool,  dataset_train_percent:float, dataset_name=None):
        # Здесь и далее все строчки со временем можешь удалить или использовать, если планируется вывод пользователю затраченного времени
        t_start = time.time()
        dataset_name = dataset_name or 'dataset'
        texts_folder = f'{images_folder}_texts'
        progress_signal.emit("Размечаю...")
        QCoreApplication.processEvents()
        t11 = time.time()
        zero_shot_folder_detection(images_folder, classes, texts_folder, min_confidence=0.005)
        t12 = time.time()
        progress_signal.emit("Разметка завершена!")
        QCoreApplication.processEvents()

        if enable_sementation:
            progress_signal.emit("Сегментирую...")
            QCoreApplication.processEvents()
            t21 = time.time()
            sam_folder_segmentation(images_folder, texts_folder)
            t22 = time.time()
            progress_signal.emit("Сегментация завершена!")
            QCoreApplication.processEvents()

        if enable_augmentation:
            progress_signal.emit("Аугментирую...")
            QCoreApplication.processEvents()
            t31 = time.time()
            augmentate_it(dir_name_images=images_folder, dir_name_textes = texts_folder)
            t32 = time.time()
            progress_signal.emit("Аугментация завершена!")
            QCoreApplication.processEvents()
        if enable_dataset:
            progress_signal.emit("Собираю датасет...")
            QCoreApplication.processEvents()
            create_empty_dataset(dataset_name)
            fill_the_dataset(dataset_name, images_folder, texts_folder, dataset_train_percent)
            create_yaml(dataset_name, list(range(len(classes))), classes)
            progress_signal.emit("Готово!")
            QCoreApplication.processEvents()
        QCoreApplication.processEvents()

    def add_class_field(self):
        row = QHBoxLayout()
        label = QLabel(f"Класс {self.fields_layout.count() + 1}:")
        line_edit = QLineEdit()
        self.classname_list.append(line_edit)
        row.addWidget(label)
        row.addWidget(line_edit)

        container = QWidget()
        container.setLayout(row)

        self.fields_layout.addWidget(container)


    def get_dataset(self):
        if os.path.exists("dataset"):
              shutil.rmtree("dataset")
        classes = [field.text() for field in self.classname_list if field.text()]
        enable_segm = self.enable_segm_checkbox.isChecked()
        enable_augm = self.enable_augm_checkbox.isChecked()

        # создаём поток и передаём все аргументы
        self.thread = WorkerThread(
            self.non_coco_mode,
            "photos", classes, enable_segm, enable_augm, True, 0.8, "dataset"
        )

        self.thread.progress.connect(self.overlay.set_message)

        self.thread.start()
        self.overlay.exec()



class ImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Dataset Editor")
        self.images = []
        self.image_paths = []
        self.current_image_index = -1
        self.image = None
        self.original_image = None
        self.selected_color = QColor("#ff0000")
        self.angle = 0
        self.scale = 1.0
        self.brightness = 0
        self.contrast = 1.0
        self.blur = 0
        self.rectangles = []
        self.is_drawing = False
        self.start_point = None
        self.temp_rect = None
        self.yolo_labels = []
        self.current_class = 0
        self.initUI()
        if os.path.exists("dataset"):
              shutil.rmtree("dataset")
        result_folder = "handmade"
        if os.path.exists(result_folder):
                shutil.rmtree(result_folder)

    def initUI(self):
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 10px;
                background: #e9e9e9;
                border-radius: 5px;
            }
            QSlider::sub-page:horizontal {
                background: #1f78d1;
                border-radius: 5px;
            }
            QSlider::add-page:horizontal {
                background: transparent;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #bfbfbf;
                border: 1px solid #9f9f9f;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QPushButton {
                background-color: gray;   /* голубая кнопка */
                color: white;
                border-radius: 5px;
                padding: 8px 8px;
            }
            QPushButton:pressed {
                background-color: darkgray;
            }
        """)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        # Сцена
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.image_item = None
        main_layout.addWidget(self.view)
        # Панель управления
        control_panel = QVBoxLayout()

        file_working_panel = QHBoxLayout()

        # Кнопка загрузки папки
        self.load_btn = QPushButton("Load Folder")
        self.load_btn.setStyleSheet("""
                            QPushButton {
                                background-color: #4169E1;   /* голубая кнопка */
                                color: white;
                                border-radius: 5px;
                                padding: 10px 10px;
                            }
                            QPushButton:pressed {
                                background-color: #003366;
                            }
                        """)
        self.load_btn.clicked.connect(self.load_folder)
        file_working_panel.addWidget(self.load_btn)
        # Сохранение итогового изображения
        self.save_btn = QPushButton("Save Image")
        self.save_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4169E1;   /* голубая кнопка */
                        color: white;
                        border-radius: 5px;
                        padding: 10px 10px;
                    }
                    QPushButton:pressed {
                        background-color: #003366;
                    }
                """)
        self.save_btn.clicked.connect(self.save_image)
        file_working_panel.addWidget(self.save_btn)

        control_panel.addLayout(file_working_panel)
        # Угол поворота
        self.angle_label = QLabel("Rotation: 0°")
        control_panel.addWidget(self.angle_label)
        self.angle_slider = QSlider(Qt.Horizontal)

        self.angle_slider.setRange(-180, 180)
        self.angle_slider.valueChanged.connect(self.update_image)
        control_panel.addWidget(self.angle_slider)

        # Яркости и контрастность
        self.brightness_label = QLabel("Brightness: 0")
        control_panel.addWidget(self.brightness_label)
        self.brightness_slider = QSlider(Qt.Horizontal)

        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.valueChanged.connect(self.update_image)
        control_panel.addWidget(self.brightness_slider)

        self.contrast_label = QLabel("Contrast: 1.0")
        control_panel.addWidget(self.contrast_label)
        self.contrast_slider = QSlider(Qt.Horizontal)

        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_image)
        control_panel.addWidget(self.contrast_slider)
        # Размытие
        self.blur_label = QLabel("Blur: 0")
        control_panel.addWidget(self.blur_label)
        self.blur_slider = QSlider(Qt.Horizontal)

        self.blur_slider.setRange(0, 20)
        self.blur_slider.valueChanged.connect(self.update_image)
        control_panel.addWidget(self.blur_slider)

        label = QLabel("Augmentation")
        control_panel.addWidget(label)

        # Создания мозаики
        self.mosaic_btn = QPushButton("Create Mosaic")
        self.mosaic_btn.clicked.connect(self.create_mosaic)
        control_panel.addWidget(self.mosaic_btn)

        self.flip_image_btn = QPushButton("Flip image")
        self.flip_image_btn.clicked.connect(self.flip_image)
        control_panel.addWidget(self.flip_image_btn)

        self.bgr_to_hsv_btn = QPushButton("Convert to HSV")
        self.bgr_to_hsv_btn.clicked.connect(self.bgr_to_hsv)
        control_panel.addWidget(self.bgr_to_hsv_btn)

        # Кнопка преобразования в ч/б
        self.bw_btn = QPushButton("Convert to Grayscale")
        self.bw_btn.clicked.connect(self.convert_to_bw)
        control_panel.addWidget(self.bw_btn)

        # Кнопка для автоаугментации
        self.auto_augmentate_button = QPushButton("Apply Autoaugmentation")
        self.auto_augmentate_button.clicked.connect(self.auto_augmentate)
        control_panel.addWidget(self.auto_augmentate_button)

        self.reset_btn = QPushButton("Reset Image")
        self.reset_btn.clicked.connect(self.reset_image)
        control_panel.addWidget(self.reset_btn)

        label = QLabel("Auto Dataset")

        control_panel.addWidget(label)

        auto_panel = QHBoxLayout()

        # Вызов окна настроек для авторазметки/сегментации и аугментации
        self.call_autoaug_button = QPushButton("Configure AutoDataset")
        self.call_autoaug_button.clicked.connect(self.call_autoaug)
        auto_panel.addWidget(self.call_autoaug_button)

        self.show_autodataset = QPushButton("Show Auto Dataset")
        self.show_autodataset.clicked.connect(self.show_segmented_images)
        auto_panel.addWidget(self.show_autodataset)

        control_panel.addLayout(auto_panel)

        label = QLabel("Annotate")

        control_panel.addWidget(label)

        annotation_panel = QHBoxLayout()

        # Выделение объектов
        self.rect_btn = QPushButton("Add Rectangle")
        self.rect_btn.clicked.connect(self.toggle_rectangle_mode)
        annotation_panel.addWidget(self.rect_btn)

        # Выделение новых объектов/новый класс
        self.mark_new_class_button = QPushButton("Add class")
        self.mark_new_class_button.clicked.connect(self.mark_new_class)
        annotation_panel.addWidget(self.mark_new_class_button)

        control_panel.addLayout(annotation_panel)

        main_layout.addLayout(control_panel)

        self.view.setMouseTracking(True)
        self.scene.installEventFilter(self)
        self.setFocusPolicy(Qt.StrongFocus)
        self.view.setFocusPolicy(Qt.NoFocus)
        self.setFocus()

        # Панель для автосегментации/аугментации
        auto_aug_panel = QVBoxLayout()

        self.add_class_button = QPushButton("Show Manual dataset")
        self.add_class_button.clicked.connect(self.show_with_mark)
        control_panel.addWidget(self.add_class_button)

    def auto_augmentate(self):
        if os.path.exists("handmade"):
            augmentate_it(Path("handmade")/Path("images"), Path("handmade")/Path("labels"))
        augmentate_it("photos")

    def choose_color(self):
        color = QColorDialog.getColor(self.selected_color, self, "Выбор цвета")
        if color.isValid():
            self.selected_color = color
            print(f"Выбранный цвет: {self.selected_color.name()}")

    def show_segmented_images(self):
        if(os.path.exists("dataset") == False):
            QMessageBox.information(None, "Warning", "AutoDataset is empty you should create it")
            return
        images = sorted([
            os.path.join(Path("dataset")/"train"/"images", f)
            for f in os.listdir(Path("dataset")/"train"/"images")
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.HEIC'))
        ])

        classes = sorted([
            os.path.join(Path("dataset")/"train"/"labels", f)
            for f in os.listdir(Path("dataset")/"train"/"labels")
            if f.lower().endswith(('.txt'))
        ])

        viewer = SegmentedViewerDialog(images, classes)
        viewer.exec()

    def show_with_mark(self):
        if(os.path.exists("dataset") == False):
            QMessageBox.information(None, "Warning", "Handmade dataset is empty you should create it")
            return

        images = [
            os.path.join(Path("handmade")/"images", f)
            for f in os.listdir(Path("handmade")/"images")
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.HEIC'))
        ]

        classes = [
            os.path.join(Path("handmade")/"labels", f)
            for f in os.listdir(Path("handmade")/"labels")
            if f.lower().endswith(('.txt'))
        ]

        viewer = ResultViewerDialog(images, classes)
        viewer.exec()

    def mark_new_class(self):
        self.choose_color()
        self.current_class += 1

    def load_folder(self):
        # Диалог выбора: папка или архив
            options = QFileDialog.Options()
            folder = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)

            if not folder:  # если папка не выбрана, попробуем файл
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Folder or ZIP Archive",
                    "",
                    "Images Folder or ZIP (*.zip);;All Files (*)",
                    options=options,
                )
                if not file_path:
                    return

                if file_path.lower().endswith(".zip"):
                    try:
                        temp_dir = tempfile.mkdtemp()
                        with zipfile.ZipFile(file_path, "r") as zip_ref:
                            zip_ref.extractall(temp_dir)
                        folder = temp_dir
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Не удалось распаковать архив:\n{e}")
                        return
                else:
                    QMessageBox.warning(self, "Warning", "Выберите папку или ZIP архив.")
                    return

        # 📂 Папка назначения внутри проекта
            photos_dir = os.path.join(os.getcwd(), "photos")

            # если папка уже есть — очистим её
            if os.path.exists(photos_dir):
                shutil.rmtree(photos_dir)
            os.makedirs(photos_dir, exist_ok=True)

            # Копируем все изображения в photos_dir
            self.image_paths = []
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.HEIC')):
                    src = os.path.join(folder, f)
                    dst = os.path.join(photos_dir, f)
                    shutil.copy2(src, dst)
                    self.image_paths.append(dst)


            self.images = []
            self.rectangles = []
            self.current_image_index = -1

            if self.image_paths:
                self.current_image_index = 0
                self.load_current_image()

    def call_autoaug(self):
        dialog = MyDialog()
        dialog.exec()

    def load_current_image(self):
        if 0 <= self.current_image_index < len(self.image_paths):
            self.image = cv2.imread(self.image_paths[self.current_image_index])
            self.original_image = self.image.copy()
            self.rectangles = []
            self.scene.clear()
            self.image_item = None
            self.reset_params()
            self.update_image()

    def reset_params(self):
        self.angle = 0
        self.scale = 1.0
        self.brightness = 0
        self.contrast = 1.0
        self.blur = 0
        self.angle_slider.setValue(0)
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.blur_slider.setValue(0)
        self.blur_label.setText("Blur: 0")

    def rotate_yolo_boxes(self, angle):
        h, w, _ = self.image.shape
        cx, cy = w / 2, h / 2

        # та же матрица, что и для картинки
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)

        new_boxes = []
        for line in self.yolo_labels:
            cls, x, y, bw, bh = line.strip().split()
            x, y, bw, bh = map(float, (x, y, bw, bh))

            # переводим в пиксели
            box_w, box_h = bw * w, bh * h
            box_x, box_y = x * w, y * h

            half_w, half_h = box_w / 2, box_h / 2
            corners = np.array([
                [box_x - half_w, box_y - half_h],
                [box_x + half_w, box_y - half_h],
                [box_x + half_w, box_y + half_h],
                [box_x - half_w, box_y + half_h]
            ])

            # применяем ту же матрицу, что к картинке
            ones = np.ones((corners.shape[0], 1))
            corners_hom = np.hstack([corners, ones])
            rotated = (M @ corners_hom.T).T

            # находим новый axis-aligned bbox
            xmin, ymin = rotated[:, 0].min(), rotated[:, 1].min()
            xmax, ymax = rotated[:, 0].max(), rotated[:, 1].max()

            # обратно в YOLO
            new_x = (xmin + xmax) / 2 / w
            new_y = (ymin + ymax) / 2 / h
            new_w = (xmax - xmin) / w
            new_h = (ymax - ymin) / h

            new_boxes.append(f"{cls} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}")

        self.yolo_labels = new_boxes

    def update_image(self):
        if(os.path.exists("photos") == False):
            QMessageBox.information(None, "Warning", "Please, load your dataset")
            return
        if self.image is None:
            return

        image = self.original_image.copy()

        self.brightness = self.brightness_slider.value()
        self.contrast = self.contrast_slider.value() / 100.0
        self.brightness_label.setText(f"Brightness: {self.brightness}")
        self.contrast_label.setText(f"Contrast: {self.contrast:.1f}")
        image = cv2.convertScaleAbs(image, alpha=self.contrast, beta=self.brightness)

        self.blur = self.blur_slider.value()
        self.blur_label.setText(f"Blur: {self.blur}")
        if self.blur > 0:
            kernel_size = max(3, self.blur * 2 + 1)
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        self.angle = self.angle_slider.value()
        self.angle_label.setText(f"Rotation Angle: {self.angle}°")
        if self.angle != 0:
            self.rotate_yolo_boxes(self.angle);
            height, width = image.shape[:2]
            center = (width / 2, height / 2)
            matrix = cv2.getRotationMatrix2D(center, -self.angle, 1.0)
            image = cv2.warpAffine(image, matrix, (width, height))

        self.image = image
        self.display_image(self.image)

    def display_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        if self.image_item:
            self.scene.removeItem(self.image_item)
        self.image_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
        self.scene.addItem(self.image_item)
        if(self.scale == 1.0):
            self.view.fitInView(self.image_item, Qt.KeepAspectRatio)
            self.view.centerOn(self.image_item)

        for rect in self.rectangles:
            self.scene.addItem(rect)

    def convert_to_bw(self):
        if self.image is not None:
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            self.update_image()

    def flip_image(self):
        self.original_image = cv2.flip(self.original_image, 1)
        self.update_image();

    def bgr_to_hsv(self):
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_shift = random.uniform(-10, 10)
        sat_mult = random.uniform(0.8, 1.2)
        val_mult = random.uniform(0.8, 1.2)

        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_mult, 0, 255)

        self.original_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        self.update_image()

    def crop_image(self):
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        x, ok = QInputDialog.getInt(self, "Crop", "Enter X coordinate:", 0, 0, width)
        if not ok:
            return
        y, ok = QInputDialog.getInt(self, "Crop", "Enter Y coordinate:", 0, 0, height)
        if not ok:
            return
        w, ok = QInputDialog.getInt(self, "Crop", "Enter width:", width//2, 1, width-x)
        if not ok:
            return
        h, ok = QInputDialog.getInt(self, "Crop", "Enter height:", height//2, 1, height-y)
        if not ok:
            return

        self.original_image = self.original_image[y:y+h, x:x+w]
        self.rectangles = []
        self.update_image()

    def reset_image(self):
        if self.image is None:
            return
        self.original_image = cv2.imread(self.image_paths[self.current_image_index])
        self.rectangles = []
        self.scene.clear()
        self.image_item = None
        self.reset_params()
        self.update_image()
        self.yolo_labels = []
        self.current_class = 0

    def create_mosaic(self):
        if not self.image_paths:
            return
        num_images, ok = QInputDialog.getInt(self, "Mosaic", "Enter number of images:", 4, 1, len(self.image_paths))
        if not ok:
            return

        images = [cv2.imread(path) for path in self.image_paths[:num_images]]
        if not images:
            return

        cols = int(math.ceil(math.sqrt(num_images)))
        rows = int(math.ceil(num_images / cols))

        max_width = max(img.shape[1] for img in images)
        max_height = max(img.shape[0] for img in images)

        mosaic = np.zeros((max_height * rows, max_width * cols, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (max_width, max_height), interpolation=cv2.INTER_AREA)
            mosaic[row*max_height:(row+1)*max_height, col*max_width:(col+1)*max_width] = img_resized

        self.original_image = mosaic
        self.rectangles = []
        self.scene.clear()
        self.image_item = None
        self.reset_params()
        self.update_image()

    def toggle_rectangle_mode(self):
        self.is_drawing = not self.is_drawing
        self.rect_btn.setText("Stop Drawing" if self.is_drawing else "Add Rectangle")
        if not self.is_drawing and self.temp_rect:
            self.scene.removeItem(self.temp_rect)
            self.temp_rect = None

    def constrain_to_image(self, point):
        if self.image is None:
            return point
        height, width = self.image.shape[:2]
        x = max(0, min(point.x(), width - 1))
        y = max(0, min(point.y(), height - 1))
        return QRectF(x, y, 0, 0).topLeft()

    def normalize_rect(self, start, end):
        x1, y1 = start.x(), start.y()
        x2, y2 = end.x(), end.y()
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        H, W, C = self.image.shape
        xMark = (abs(x2 + x1) / 2) / W
        yMark = (abs(y2 + y1) / 2) / H
        wMark = w / W
        hMark = h / H
        return (QRectF(x, y, w, h), [ self.current_class, xMark, yMark, wMark, hMark ])

    def eventFilter(self, source, event):
        if source is self.scene and self.image is not None and self.is_drawing:
            if event.type() == event.GraphicsSceneMousePress:
                self.start_point = self.constrain_to_image(event.scenePos())
                self.temp_rect = QGraphicsRectItem(QRectF(self.start_point, self.start_point))

                pen = QPen(self.selected_color)
                self.temp_rect.setPen(pen)
                self.scene.addItem(self.temp_rect)
                return True
            elif event.type() == event.GraphicsSceneMouseMove and self.start_point:
                end_point = self.constrain_to_image(event.scenePos())

                f, s = self.normalize_rect(self.start_point, end_point)

                self.temp_rect.setRect(f)
                return True
            elif event.type() == event.GraphicsSceneMouseRelease and self.start_point:
                end_point = self.constrain_to_image(event.scenePos())

                f, s = self.normalize_rect(self.start_point, end_point)

                self.yolo_labels.append(f"{s[0]} {s[1]:.6f} {s[2]:.6f} {s[3]:.6f} {s[4]:.6f}")

                rect = QGraphicsRectItem(f)

                print(self.yolo_labels)

                rect.setPen(QPen(self.selected_color))
                self.rectangles.append(rect)
                self.scene.removeItem(self.temp_rect)
                self.temp_rect = None
                self.scene.addItem(rect)
                self.start_point = None
                return True
        return super().eventFilter(source, event)

    def save_image(self):
        if self.image is None or self.current_image_index < 0 or self.current_image_index >= len(self.image_paths):
            return

        result_folder = "handmade"
        res_img_folder = Path("handmade")/"images"
        res_labels_folder = Path("handmade")/"labels"

        os.makedirs(result_folder, exist_ok = True)
        os.makedirs(res_img_folder, exist_ok = True)
        os.makedirs(res_labels_folder, exist_ok = True)

        file_name = f"image_{int(time.time())}"
        cv2.imwrite(Path("handmade")/"images"/f"{file_name}.jpg", self.image)

        with open(Path("handmade")/"labels"/f"{file_name}.txt", "w") as f:
            for txt in self.yolo_labels:
                f.write(txt + "\n")

        self.yolo_labels = []
        self.current_class = 0
        self.display_image(self.original_image)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_W) and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
        elif event.key() in (Qt.Key_Down, Qt.Key_S) and self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.load_current_image()
        self.yolo_labels = []
        self.current_class = 0

if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = ImageEditor()
    editor.resize(1000, 600)
    editor.show()
    sys.exit(app.exec_())
