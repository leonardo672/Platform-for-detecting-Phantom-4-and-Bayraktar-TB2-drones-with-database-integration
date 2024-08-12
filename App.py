import sys
import cv2
import numpy as np
import torch
import os
import pyodbc
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QMainWindow, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QLabel, QGridLayout, QScrollArea, QMessageBox, QDialog,
    QLineEdit
)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QImage, QPalette, QBrush
from PyQt5.QtCore import QObject, pyqtSignal, Qt
import pandas as pd
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtWidgets import QDialog, QGridLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QMessageBox, QVBoxLayout, QHeaderView


class Overviewofmodelform(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Обзор силы модели')
        self.initUI()

    def initUI(self):
         # Set the size of the main window
        self.setGeometry(50, 50, 1500, 1200)

        # Load background image
        bg_image_path = "images/is-machine-learning-hard-a-guide-to-getting-started-scaled-1-scaled.jpeg"  # Specify the path to your background image
        original_pixmap = QPixmap(bg_image_path)
        # Resize the background image to fit the window
        pixmap = original_pixmap.scaled(1500, 800)

        # Create brush with scaled pixmap
        brush = QBrush(pixmap)

        # Create palette and set the background brush
        palette = self.palette()
        palette.setBrush(QPalette.Background, brush)

        # Set the palette to the widget
        self.setPalette(palette)

        # Set the size of the main window to match the background image size
        self.setFixedSize(pixmap.size())

        layout = QVBoxLayout(self)

      #  layout = QVBoxLayout(self)
            #  self.bayraktar_button.setFixedSize(180, 70)

        #.....
        self.confusion_matrix_button = QPushButton('Матрица путаницы', self)
        self.confusion_matrix_button.setGeometry(30, 6, 220, 60)
        self.confusion_matrix_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.confusion_matrix_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.confusion_matrix_button.clicked.connect(self.open_confusion_matrix_image)
        #.....
        self.F1_curve_button = QPushButton('Кривая F1', self)
        self.F1_curve_button.setGeometry(260, 6, 150, 60)
        self.F1_curve_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.F1_curve_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.F1_curve_button.clicked.connect(self.open_F1_curve_image)
        #.....
        self.labels_button = QPushButton('Метки', self)
        self.labels_button.setGeometry(420, 6, 150, 60)
        self.labels_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.labels_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.labels_button.clicked.connect(self.open_labels_image)
         #.....
        self.labels_correlogram_button = QPushButton('Коррелограмма \n меток', self)
        self.labels_correlogram_button.setGeometry(585, 6, 200, 60)
        self.labels_correlogram_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.labels_correlogram_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.labels_correlogram_button.clicked.connect(self.open_labels_correlogram_image)
        #.....
        self.P_curve_button = QPushButton('Кривая P', self)
        self.P_curve_button.setGeometry(800, 6, 150, 60)
        self.P_curve_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.P_curve_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.P_curve_button.clicked.connect(self.open_P_curve_image)
        #.....
        self.PR_curve_button = QPushButton('Кривая PR', self)
        self.PR_curve_button.setGeometry(960, 6, 150, 60)
        self.PR_curve_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.PR_curve_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.PR_curve_button.clicked.connect(self.open_PR_curve_image)
        #.....
        self.R_curve_button = QPushButton('Кривая R', self)
        self.R_curve_button.setGeometry(1120, 6, 150, 60)
        self.R_curve_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.R_curve_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.R_curve_button.clicked.connect(self.open_R_curve_image)
        #.....
        self.results_curve_button = QPushButton('Результаты', self)
        self.results_curve_button.setGeometry(1280, 6, 150, 60)
        self.results_curve_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.results_curve_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.results_curve_button.clicked.connect(self.open_results_image)

        self.setLayout(layout)
       # layout.addWidget(self.phantom_button)

        #layout.setAlignment(Qt.AlignTop)  # Align buttons to the top of the layout

    def open_confusion_matrix_image(self):
        folder_path = "images/model/confusion_matrix"
        self.open_folder_images(folder_path)

    def open_F1_curve_image(self):
        folder_path = "images/model/F1_curve"
        self.open_folder_images(folder_path)

    def open_labels_image(self):
        folder_path = "images/model/labels"
        self.open_folder_images(folder_path)

    def open_labels_correlogram_image(self):
        folder_path = "images/model/labels_correlogram"
        self.open_folder_images(folder_path)

    def open_P_curve_image(self):
        folder_path = "images/model/P_curve"
        self.open_folder_images(folder_path)

    def open_PR_curve_image(self):
        folder_path = "images/model/PR_curve"
        self.open_folder_images(folder_path)

    def open_R_curve_image(self):
        folder_path = "images/model/R_curve"
        self.open_folder_images(folder_path)

    def open_results_image(self):
        folder_path = "images/model/results"
        self.open_folder_images(folder_path)

    def open_folder_images(self, folder_path):
        # Clear existing layout
        for i in reversed(range(self.layout().count())):
            self.layout().itemAt(i).widget().setParent(None)

        # Create a new layout for the images
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow scroll area to resize widget
        scroll_area.setFixedSize(1400, 600)  # Adjust the size as needed
        scroll_area.move(200, 25)  # Move the scroll area to the specified position

        self.layout().addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        scroll_widget.setStyleSheet("background-color: #345C7F;")

        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setHorizontalSpacing(20)  # Adjust horizontal spacing between images
        grid_layout.setVerticalSpacing(20)    # Adjust vertical spacing between images
        grid_layout.setColumnStretch(1, 1)     # Stretch the columns to evenly distribute available space

        # Load images from the specified folder path
        images = os.listdir(folder_path)
        num_columns = 3  # Number of columns in the grid layout
        for index, image_name in enumerate(images):
            row = index // num_columns
            col = index % num_columns
            image_path = os.path.join(folder_path, image_name)
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(1200, 600)  # Adjust the size here
            image_label.setPixmap(pixmap)
            grid_layout.addWidget(image_label, row, col)

        self.adjustSize()  # Adju



class StructureOfObjectsForm(QWidget):   # (320, 9, 180, 70)
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Структура объектов')
      #  self.setGeometry(200, 200, 800, 600)
        self.initUI()
       # layout = QVBoxLayout()

        # Button to open Bayraktar images
       # bayraktar_button = QPushButton('Open Bayraktar Images', self)
       # bayraktar_button.clicked.connect(self.open_bayraktar_images)
       # bayraktar_button.setGeometry(200, 10, 200, 70)
       # bayraktar_button.setFixedSize(180, 70)
       # layout.addWidget(bayraktar_button)

      #  bayraktar_button.setStyleSheet(
       #     "QPushButton { background-color: white; color: black; "
     #       "border: 2px solid gray; border-radius: 10px; max-width: 150px;"
      #      "min-width: 100px; min-height: 30px; }"
      #      "QPushButton:hover { background-color: lightgray; }"
      #  )
      #  layout.addWidget(bayraktar_button)

        # Button to open Phantom images
      #  phantom_button = QPushButton('Open Phantom Images', self)
        # phantom_button.clicked.connect(self.open_phantom_images)
       # phantom_button.setFixedSize(150, 30)
      #  bayraktar_button.setGeometry(400, 10, 200, 70)
      #  layout.addWidget(phantom_button)
     #   phantom_button.setStyleSheet(
      #      "QPushButton { background-color: white; color: black; "
     #       "border: 2px solid gray; border-radius: 10px; "
      #      "min-width: 100px; min-height: 30px; }"
      #      "QPushButton:hover { background-color: lightgray; }"
      #  )
        #layout.addWidget(phantom_button)

        #self.setLayout(layout)

       # self.child_form_button = QPushButton('Real Time Database', self)
       # self.child_form_button.setGeometry(720, 9, 230, 70)
      #  self.child_form_button.setStyleSheet("background-color: white; color: black;")
       # self.child_form_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
       # self.child_form_button.clicked.connect(self.open_child_form)

    def initUI(self):
         # Set the size of the main window
        self.setGeometry(100, 100, 800, 700)

        # Load background image
        bg_image_path = "images/LOP2.png"  # Specify the path to your background image
        original_pixmap = QPixmap(bg_image_path)
        # Resize the background image to fit the window
        pixmap = original_pixmap.scaled(800, 700)

        # Create brush with scaled pixmap
        brush = QBrush(pixmap)

        # Create palette and set the background brush
        palette = self.palette()
        palette.setBrush(QPalette.Background, brush)

        # Set the palette to the widget
        self.setPalette(palette)

        # Set the size of the main window to match the background image size
        self.setFixedSize(pixmap.size())

        layout = QVBoxLayout(self)

      #  layout = QVBoxLayout(self)
            #  self.bayraktar_button.setFixedSize(180, 70)
        self.bayraktar_button = QPushButton('Структура Bayraktar_TB2', self)
        self.bayraktar_button.setGeometry(130, 6, 290, 60)
        self.bayraktar_button.setStyleSheet(
            "QPushButton {"
            "   background-color: gray; "
            "   color: black; margin-bottom: 0px; margin-top: 0px; "
            "   border: 2px solid gray; "
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.bayraktar_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.bayraktar_button.clicked.connect(self.open_bayraktar_images)

        # Button to open Phantom images
        self.phantom_button = QPushButton('Структура Phantom_4', self)
        self.phantom_button.setGeometry(450, 6, 245, 60)
        self.phantom_button.setStyleSheet(
            "QPushButton {"
            "   background-color: gray; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.phantom_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.phantom_button.clicked.connect(self.open_phantom_images)

        self.setLayout(layout)
       # layout.addWidget(self.phantom_button)

        #layout.setAlignment(Qt.AlignTop)  # Align buttons to the top of the layout

    def open_bayraktar_images(self):
        folder_path = "images/Bayraktar"
        self.open_folder_images(folder_path)

    def open_phantom_images(self):
        folder_path = "images/Phantom"
        self.open_folder_images(folder_path)

    def open_folder_images(self, folder_path):
        # Clear existing layout
        for i in reversed(range(self.layout().count())):
            self.layout().itemAt(i).widget().setParent(None)

        # Create a new layout for the images
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow scroll area to resize widget

        scroll_area.setFixedSize(800, 500)  # Adjust the size as needed

        self.layout().addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        scroll_widget.setStyleSheet("background-color: #345C7F;")

        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setHorizontalSpacing(20)  # Adjust horizontal spacing between images
        grid_layout.setVerticalSpacing(20)    # Adjust vertical spacing between images
        grid_layout.setColumnStretch(1, 1)     # Stretch the columns to evenly distribute available space

        # Load images from the specified folder path
        images = os.listdir(folder_path)
        num_columns = 3  # Number of columns in the grid layout
        for index, image_name in enumerate(images):
            row = index // num_columns
            col = index % num_columns
            image_path = os.path.join(folder_path, image_name)
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(600, 600)  # Adjust the size here
            image_label.setPixmap(pixmap)
            grid_layout.addWidget(image_label, row, col)

        self.adjustSize()  # Adjust the size of the window to fit the images

class ImageDisplayForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('База данных проанализированных изображений')
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow scroll area to resize widget
        layout.addWidget(scroll_area)

        # Create a widget for the scroll area
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)

        # Set black background color for the scroll widget
        scroll_widget.setStyleSheet("background-color: #345C7F;")

        # Layout for the images
        grid_layout = QGridLayout(scroll_widget)

        # Load images from folder
        folder_path = "images/P"
        images = os.listdir(folder_path)
        row, col = 0, 0
        for image_name in images:
            image_path = os.path.join(folder_path, image_name)
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(300, 300)  # Adjust the size here
            image_label.setPixmap(pixmap)
            grid_layout.addWidget(image_label, row, col)
            col += 1
            if col == 3:  # Max 3 images per row
                col = 0
                row += 1

        self.setLayout(layout)

class ResultsWindow(QMainWindow):
    def __init__(self, image_path=None, folder_path=None, video_path=None, model_path=None, image_size=(400, 400)):
        super().__init__()
        self.setWindowTitle('Результаты анализа')
        self.image_size = image_size

        layout = QVBoxLayout()

        # Load model if provided
        if model_path:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        else:
            self.model = None

        # Analyze single image if provided
        if image_path:
            self.analyze_single_image(image_path, layout)

        # Analyze images in a folder if provided
        if folder_path:
            self.analyze_images_in_folder(folder_path, layout)

        # Analyze video if provided
        if video_path:
            self.analyze_video(video_path, layout)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def analyze_single_image(self, image_path, layout):
        # Perform image analysis
        results = self.model(image_path) if self.model else None

        # Display results
        self.display_results(results, layout)

    def analyze_images_in_folder(self, folder_path, layout):
        # Analyze images in the specified folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                results = self.model(image_path) if self.model else None
                self.display_results(results, layout)

    def analyze_video(self, video_path, layout):
        # Perform object detection on the video
        detect_and_save_objects(video_path)

    def display_results(self, results, layout):
        if results:
            # Display text results
            text_results = str(results)
            text_label = QLabel(text_results, self)
            layout.addWidget(text_label)

            # Display image with detections
            img_array = np.squeeze(results.render())
            img_array_resized = cv2.resize(img_array, self.image_size)
            height, width, channel = img_array_resized.shape
            bytesPerLine = 3 * width
            qImg = QImage(img_array_resized.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            img_label = QLabel(self)
            img_label.setPixmap(pixmap)
            layout.addWidget(img_label)

class VideoAnalyzer(QObject):
    video_analysis_finished = pyqtSignal()

    def detect_and_save_objects(self, input_path, new_width=640, new_height=480):
        # Load the YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='The_last_2_of_Phantom_and_bayre/exp/weights/last.pt', force_reload=True)

        # Open the video file for reading
        cap = cv2.VideoCapture(input_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            frame = cv2.resize(frame, (new_width, new_height))

            # Make detections on the frame
            results = model(frame)

            # Extract the detected objects from results (you need to adjust this part based on your YOLO model output)
            detected_objects = results.xyxy[0]

            # Save detected objects to the specified path
            for obj in detected_objects:
                x1, y1, x2, y2, confidence, class_id = obj
                object_image = frame[int(y1):int(y2), int(x1):int(x2)]

                # Save the detected object to the specified path
                object_filename = f"Our_DataBase/object_{int(x1)}_{int(y1)}.jpg"
                cv2.imwrite(object_filename, object_image)

            # Display the frame with YOLO detections (optional)
            cv2.imshow('Видеоанализ', np.squeeze(results.render()))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture object
        cap.release()

        # Destroy any OpenCV windows
        cv2.destroyAllWindows()

        # Emit signal to indicate video analysis is finished
        self.video_analysis_finished.emit()

class ImageDisplayForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('База данных проанализированных изображений')
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow scroll area to resize widget
        layout.addWidget(scroll_area)

        # Create a widget for the scroll area
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)

        # Set black background color for the scroll widget
        scroll_widget.setStyleSheet("background-color: #345C7F;")

        # Layout for the images
        grid_layout = QGridLayout(scroll_widget)

        # Load images from folder
        folder_path = "images/P"
        images = os.listdir(folder_path)
        row, col = 0, 0
        for image_name in images:
            image_path = os.path.join(folder_path, image_name)
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(300, 300)  # Adjust the size here
            image_label.setPixmap(pixmap)
            grid_layout.addWidget(image_label, row, col)
            col += 1
            if col == 3:  # Max 3 images per row
                col = 0
                row += 1

        self.setLayout(layout)


class ImagesChildForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Базе данных реального времени')
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow scroll area to resize widget
        layout.addWidget(scroll_area)

        # Create a widget for the scroll area
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)

        # Set black background color for the scroll widget
        scroll_widget.setStyleSheet("background-color: #345C7F;")

        # Layout for the images
        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setHorizontalSpacing(20)  # Adjust horizontal spacing between images
        grid_layout.setVerticalSpacing(20)    # Adjust vertical spacing between images
        grid_layout.setColumnStretch(1, 1)     # Stretch the columns to evenly distribute available space


        # Load images from folder
        folder_path = "Our_DataBase"
        images = os.listdir(folder_path)
        num_columns = 3  # Number of columns in the grid layout
        for index, image_name in enumerate(images):
            row = index // num_columns
            col = index % num_columns
            image_path = os.path.join(folder_path, image_name)
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaled(300, 300)  # Adjust the size here
            image_label.setPixmap(pixmap)
            grid_layout.addWidget(image_label, row, col)

        self.setLayout(layout)

    def display_images(self, layout):
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(self.folder_path, filename)
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaledToWidth(600)  # Adjust the width as needed
                    label = QLabel()
                    label.setPixmap(pixmap)
                    layout.addWidget(label)

    def open_image_display_form(self):
        # Open image display form
        self.image_display_form = ImagesChildForm(self.folder_path)
        self.image_display_form.show()


class AccessDatabaseForm(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('База данных обнаружений')
        self.setGeometry(200, 200, 800, 600)
        self.setStyleSheet("background-color: #f0f0f0;")  # Set background color

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        # Create a table to display Access database data
        self.table_widget = QTableWidget()
        self.table_widget.setStyleSheet("background-color: #B0BBC1;")  # Set table background color
        layout.addWidget(self.table_widget)

        # Load Access database data
        self.load_access_data()

        self.central_widget.setLayout(layout)

    def load_access_data(self):
        # Connect to the Access database
        db_path = r'Ac\Homam.accdb'
        conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};'
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Execute a query to retrieve data from the Detections table
        cursor.execute("SELECT * FROM Detections")
        rows = cursor.fetchall()

        # Get column names
        columns = [column[0] for column in cursor.description]

        # Display data in the table
        self.table_widget.setRowCount(len(rows))
        self.table_widget.setColumnCount(len(columns))
        self.table_widget.setHorizontalHeaderLabels(columns)
        self.table_widget.horizontalHeader().setStyleSheet("background-color: #d9d9d9;")  # Set header background color
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # Stretch columns to fit content

        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                cell_value = str(value)
                item = QTableWidgetItem(cell_value)
                self.table_widget.setItem(i, j, item)

        # Close the connection
        conn.close()



class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Вход или регистрация')
        self.setGeometry(200, 200, 1400, 600)  # Adjusted window size

        layout = QGridLayout()  # Use a grid layout

        # Set background color and text color
        self.setStyleSheet("background-color:#104675; color: #f9f9f9;")

        # Create login widgets
        login_username_label = QLabel('Имя пользователя:')
        login_password_label = QLabel('Пароль:')
        login_username_edit = QLineEdit()
        login_password_edit = QLineEdit()
        login_password_edit.setEchoMode(QLineEdit.Password)
        login_button = QPushButton('Вход')

        # Apply styling to labels and buttons
        labels = [login_username_label, login_password_label]
        for label in labels:
            label_style = "font: 25px 'Times New Roman'; color: #f9f9f9; margin-bottom: 5px;"
            label.setStyleSheet(label_style)

        login_username_edit.setStyleSheet("font-size: 25px; padding: 8px; font: bold 25px 'Times New Roman'; border: 2px solid #ccc; border-radius: 10px; background-color: #444; color: #f9f9f9;")
        login_password_edit.setStyleSheet("font-size: 25px; padding: 8px; font: bold 25px 'Times New Roman'; border: 2px solid #ccc; border-radius: 10px; background-color: #444; color: #f9f9f9;")

        login_username_edit.setFixedWidth(230)
        login_password_edit.setFixedWidth(230)

        login_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #858585; "  # Button background color
            "   color: white; "
            "   padding: 15px 30px; "
            "   border: none; "
            "   border-radius: 10px; "
            "   font-family: 'Times New Roman'; "  # Font family
            "   font-weight: bold; "  # Font weight
            "   font-size: 25px; "  # Font size
            "}"
            "QPushButton:hover {"
            "   background-color: #606060; "  # Hover background color
            "}"
        )

        layout.addWidget(login_username_label, 0, 0)
        layout.addWidget(login_username_edit, 0, 1)
        layout.addWidget(login_password_label, 1, 0)
        layout.addWidget(login_password_edit, 1, 1)
        layout.addWidget(login_button, 2, 1)

        # Create a horizontal layout
        registration_layout = QHBoxLayout()

         # Create registration widgets
        register_username_label = QLabel('Имя пользователя:')
        register_email_label = QLabel('Электронная почта:')
        register_phone_label = QLabel('Номер телефона:')
       # registration_layout.addSpacing(6)
        register_password_label = QLabel('Пароль:')
      #  registration_layout.addSpacing(6)
        register_username_edit = QLineEdit()
        register_email_edit = QLineEdit()
        register_phone_edit = QLineEdit()
        register_password_edit = QLineEdit()
        register_password_edit.setEchoMode(QLineEdit.Password)
        register_button = QPushButton('Регистрация')

        # Add widgets to the horizontal layout
        registration_layout.addWidget(register_username_label)
        registration_layout.addWidget(register_username_edit)
        registration_layout.addWidget(register_email_label)
        registration_layout.addWidget(register_email_edit)
        registration_layout.addWidget(register_phone_label)
        registration_layout.addWidget(register_phone_edit)
        registration_layout.addWidget(register_password_label)
        registration_layout.addWidget(register_password_edit)
        registration_layout.addWidget(register_button)


# Create a horizontal layout
#layout = QHBoxLayout()

# Create widgets
#register_username_label = QLabel('Username:')
#register_username_edit = QLineEdit()

# Add widgets to the layout
#layout.addWidget(register_username_label)
#layout.addWidget(register_username_edit)

# Set the layout for the dialog
#dialog.setLayout(layout)

#dialog.show()

      #  layout = QVBoxLayout(dialog)

# Create a QLabel
#label = QLabel("Position me!")
#label.setStyleSheet("background-color: #668798; color: #070102; font-size: 20px;")

# Set the position of the label within the layout
#layout.addWidget(label)
#layout.setContentsMargins(0, 30, 0, 0)

        # Apply styling to registration labels and input fields
        register_labels = [register_username_label, register_email_label, register_phone_label, register_password_label]
        for label in register_labels:
            label_style = "font: 25px 'Times New Roman'; color: #f9f9f9; margin-bottom: 5px;"
            label.setStyleSheet(label_style)

        register_edit_style = "font: bold 25px 'Times New Roman'; padding: 8px; border: 2px solid #ccc; border-radius: 10px; background-color: #444; color: #f9f9f9;"
        register_username_edit.setStyleSheet(register_edit_style)
        register_email_edit.setStyleSheet(register_edit_style)
        register_phone_edit.setStyleSheet(register_edit_style)
        register_password_edit.setStyleSheet(register_edit_style)

        register_username_edit.setFixedWidth(300)
        register_email_edit.setFixedWidth(300)
        register_phone_edit.setFixedWidth(300)
        register_password_edit.setFixedWidth(300)

        register_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #858585; "  # Button background color
            "   color: white; "
            "   padding: 15px 30px; "
            "   border: none; "
            "   border-radius: 10px; "
            "   font-family: 'Times New Roman'; "  # Font family
            "   font-weight: bold; "  # Font weight
            "   font-size: 25px; "  # Font size
            "}"
            "QPushButton:hover {"
            "   background-color: #606060; "  # Hover background color
            "}"
        )

        layout.addWidget(register_username_label, 0, 2)
        layout.addWidget(register_username_edit, 0, 3)
        layout.addWidget(register_email_label, 1, 2)
        layout.addWidget(register_email_edit, 1, 3)
        layout.addWidget(register_phone_label, 2, 2)
        layout.addWidget(register_phone_edit, 2, 3)
        layout.addWidget(register_password_label, 3, 2)
        layout.addWidget(register_password_edit, 3, 3)
        layout.addWidget(register_button, 4, 3)

        # Add the image on the right side
        pixmap = QPixmap("images/Drone-500x300-1-400x240.jpg")  # Corrected file path
        pixmap = pixmap.scaled(600, 600)  # Adjust size as needed
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label, 0, 4, 5, 1)  # Span multiple rows

        self.setLayout(layout)

        # Assign variables for login and register edit fields
        self.username_edit = login_username_edit
        self.password_edit = login_password_edit
        self.register_username_edit = register_username_edit
        self.register_password_edit = register_password_edit
        self.register_email_edit = register_email_edit
        self.register_phone_edit = register_phone_edit

        # Connect button clicks to functions
        login_button.clicked.connect(self.login)
        register_button.clicked.connect(self.register)

    def login(self):
        # Implement login logic here
        username = self.username_edit.text()
        password = self.password_edit.text()

        if not username or not password:  # Check if any field is empty
            QMessageBox.warning(self, 'Вход не выполнен', 'Пожалуйста, введите как имя пользователя, так и пароль.')
            return

        # Connect to the Access database
        connection_string = r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=Ac\Homam.accdb;'
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        try:
            # Check credentials
            cursor.execute("SELECT Username, Password FROM Users WHERE Username=? AND Password=?", (username, password))
            user = cursor.fetchone()
            if user:
                self.accept()  # Close the login window and proceed
            else:
                QMessageBox.warning(self, 'Вход не выполнен', 'неправильное имя пользователя или пароль')
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Произошла ошибка: {str(e)}')
        finally:
            # Close database connection
            cursor.close()
            conn.close()

    def register(self):
        # Implement registration logic here
        new_username = self.register_username_edit.text()
        new_password = self.register_password_edit.text()
        new_email = self.register_email_edit.text()
        new_phone = self.register_phone_edit.text()

        if not all([new_username, new_password, new_email, new_phone]):  # Check if any field is empty
            QMessageBox.warning(self, 'Регистрация не удалас', 'Пожалуйста заполните все поля.')
            return

        # Connect to the Access database
        connection_string = r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=Ac\Homam.accdb;'
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        try:
            # Check if the username already exists
            cursor.execute("SELECT Username FROM Users WHERE Username=?", (new_username,))
            existing_user = cursor.fetchone()
            if existing_user:
                QMessageBox.warning(self, 'Регистрация не удалась', 'Имя пользователя уже занято')
            else:
                # Insert new user into the database
                cursor.execute("INSERT INTO Users (Username, Password, Email, Phone_Number) VALUES (?, ?, ?, ?)", (new_username, new_password, new_email, new_phone))
                conn.commit()
                QMessageBox.information(self, 'Регистрация прошла успешно', 'Новый пользователь зарегистрирован успешно')
                # Clear registration fields after registration
                self.register_username_edit.clear()
                self.register_password_edit.clear()
                self.register_email_edit.clear()
                self.register_phone_edit.clear()
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Произошла ошибка: {str(e)}')
        finally:
            # Close database connection
            cursor.close()
            conn.close()

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
      #  self.image_display_label = QLabel()
        self.initUI()


    def initUI(self):
        # Set the size of the main window
        self.setGeometry(100, 100, 1900, 900)

        # Calculate button width based on the main window width
        button_width = int((self.width() - 100) / 7)

        # Set button heights
        button_height = 70

        # Load background image
        bg_image_path = "images/EFGwdoSkHqhIgoyJiZIyXpHtPoq9l5KYkR2uYZu-JtQ.png"  # Specify the path to your background image
        original_pixmap = QPixmap(bg_image_path)
        # Resize the background image to fit the window
        pixmap = original_pixmap.scaled(1900, 900)

        # Create brush with scaled pixmap
        brush = QBrush(pixmap)

        # Create palette and set the background brush
        palette = self.palette()
        palette.setBrush(QPalette.Background, brush)

        # Set the palette to the widget
        self.setPalette(palette)

        # Set the size of the main window to match the background image size
        self.setFixedSize(pixmap.size())

        # Create buttons for image and video analysis
        self.image_button = QPushButton('Анализ изображений', self)
        self.image_button.setGeometry(50, 9, 235, button_height)
        self.image_button.setStyleSheet("background-color: white; color: black;")
        self.image_button.setFont(QFont("Times New Roman", 14, QFont.Bold))  # Set font size and style
        self.image_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.image_button.clicked.connect(self.open_image_results_window)

        self.video_button = QPushButton('Видео анализ', self)
        self.video_button.setGeometry(300, 9, 200, button_height)
        self.video_button.setStyleSheet("background-color: white; color: black;")
        self.video_button.setFont(QFont("Times New Roman", 14, QFont.Bold))  # Set font size and style
        self.video_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.video_button.clicked.connect(self.open_video_results_window)

        self.database_button = QPushButton('База данных\nобнаружений', self)
        self.database_button.setStyleSheet("QPushButton { text-align: left; }")
        self.database_button.setGeometry(515, 9, 225, button_height)
        self.database_button.setStyleSheet("background-color: white; color: black;")
        self.database_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.database_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.database_button.clicked.connect(self.open_database_form)
       # layout.addWidget(self.database_button)
        #self.child_form_button = QPushButton('База данных реального времени', self)

        self.child_form_button = QPushButton('База данных\nреального времени', self)
        self.child_form_button.setStyleSheet("QPushButton { text-align: left; }")

        self.child_form_button.setGeometry(750, 9, 240, button_height)
        self.child_form_button.setStyleSheet("background-color: white; color: black;")
        self.child_form_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.child_form_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.child_form_button.clicked.connect(self.open_child_form)

        #self.image_display_button = QPushButton('База данных проанализированных изображений', self)
        self.image_display_button = QPushButton('База данных\nпроанализированных изображений', self)
        self.image_display_button.setStyleSheet("QPushButton { text-align: left; }")

        self.image_display_button.setGeometry(1000, 9, 380, button_height)
        self.image_display_button.setStyleSheet("background-color: white; color: black;")  # Set white background and black text
        self.image_display_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.image_display_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.image_display_button.clicked.connect(self.open_image_display_form)

       # self.structure_button = QPushButton('Object Structure', self)
     #   button1.setToolTip('This is a button for showing the structure of objects')
     #   button1.setGeometry(650, 300, 500, 100)
     #   button1.setFont(QFont("Arial", 18))
      #  button1.clicked.connect(self.show_structure_of_objects)

        self.structure_button = QPushButton('Структура объектов', self)
        self.structure_button.setGeometry(1390, 9, 230, button_height)
        self.structure_button.setStyleSheet("background-color: white; color: black;")
        self.structure_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.structure_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.structure_button.clicked.connect(self.open_structure_form)


        self.model_button = QPushButton('Обзор силы модели', self)
        self.model_button.setGeometry(1630, 9, 250, button_height)
        self.model_button.setStyleSheet("background-color: white; color: black;")
        self.model_button.setFont(QFont("Times New Roman", 14, QFont.Bold))
        self.model_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #B0BBC1; "
            "   color: black; "
            "   border: 2px solid gray; margin-bottom: 0px; margin-top: 0px;"
            "   border-radius: 10px; "
            "}"
            "QPushButton:hover {"
            "   background-color: lightgray; "
            "}"
        )
        self.model_button.clicked.connect(self.open_model_form)


        self.image_display_label = QLabel(self)
        self.image_display_label.setGeometry(100, 100, 1800, 600)  # Adjust the geometry as needed

        # layout.addWidget(self.image_display_button)

        self.setWindowTitle('Проект обнаружения дронов')
        self.show()

    def open_model_form(self):
        self.Overview_of_model_form = Overviewofmodelform()
        self.Overview_of_model_form.show()

    def open_image_results_window(self):
        # Open file dialog to select an image or a folder
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.jpg *.png)")
        file_dialog.setViewMode(QFileDialog.Detail)
        filenames, _ = file_dialog.getOpenFileNames(self, "Выберите изображение или папку", "", "Images (*.jpg *.png);;All Files (*)", options=options)

        if filenames:
            # Define paths
            model_path = 'The_last_2_of_Phantom_and_bayre/exp/weights/last.pt'
            image_size = (800, 600)

            # Open the results window for images
            self.results_window = ResultsWindow(image_path=filenames[0], folder_path=None, video_path=None, model_path=model_path, image_size=image_size)
            self.results_window.show()

    def open_video_results_window(self):
        # Open file dialog to select a video
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Videos (*.mp4 *.avi)")
        file_dialog.setViewMode(QFileDialog.Detail)
        filenames, _ = file_dialog.getOpenFileNames(self, "Выберите видео", "", "Videos (*.mp4 *.avi)", options=options)

        if filenames:
            # Perform video analysis
            self.video_analyzer = VideoAnalyzer()
            self.video_analyzer.video_analysis_finished.connect(self.open_child_form)
            self.video_analyzer.detect_and_save_objects(filenames[0])

    def open_child_form(self):
        # Open child form to show images from the specified folder
       # folder_path = "C:/Users/L/data/Our_DataBase"
        self.child_form = ImagesChildForm()
        self.child_form.show()

      #  self.image_display_form = ImageDisplayForm()
      #  self.image_display_form.show()
    def open_database_form(self):
        self.database_form = AccessDatabaseForm()
        self.database_form.show()

   # def open_database_form(self):
   #     self.database_form = ExcelDatabaseForm()
    #    self.database_form.show()

    def open_image_display_form(self):
        self.image_display_form = ImageDisplayForm()
        self.image_display_form.show()

    def open_structure_form(self):
        self.structure_of_objects_form = StructureOfObjectsForm()
        self.structure_of_objects_form.show()

    def show_structure_of_objects(self):
        self.structure_of_objects_form = StructureOfObjectsForm()
        self.structure_of_objects_form.show()

    def show_image_display(self):
        self.image_display_form = ImageDisplayForm()
        self.image_display_form.show()

    def display_images(self, folder_path):
        # Clear existing images
        self.image_display_label.clear()

        # Display images in the label
        image_files = os.listdir(folder_path)
        if image_files:
            # Create a layout to hold the images
            layout = QVBoxLayout()

            # Load and display each image
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaledToWidth(400)  # Adjust the width as needed
                    label = QLabel()
                    label.setPixmap(pixmap)
                    layout.addWidget(label)

            # Set the layout containing images to the label
            self.image_display_label.setLayout(layout)

    #def open_database_form(self):
      #  self.database_form = ExcelDatabaseForm()
      #  self.database_form.show()


# class ImagesChildForm(QWidget):

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = LoginWindow()  # Create the login window
    if login_window.exec_() == QDialog.Accepted:  # If the user successfully logs in or registers
        main_window = MyWindow()  # Create the main interface window
        main_window.show()  # Show the main interface window
        sys.exit(app.exec_())  # Exit the application when the main window is closed
