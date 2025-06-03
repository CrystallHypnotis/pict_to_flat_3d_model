import os
import sys
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLineEdit,
    QGridLayout,
    QPushButton,
    QLabel,
    QListWidget
)
import numpy as np
from scipy import spatial
from stl import mesh
import sqlite3
import datetime
from PIL import Image as Img
import numpy as np
import sys
import cv2 as cv


def sobel_compress(argv):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    file = argv

    if len(argv) < 1:
        print('Not enough parameters')
        print('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1
    # Load the image
    src = cv.imread(argv, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image: ' + argv)
        return -1

    src = cv.GaussianBlur(src, (3, 3), 0)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv.waitKey(0)
    cv.imwrite(f"{file[:-4]}_compressed{file[-4:]}", grad)

    return 0


def pict_slc(file="TEST.png", threshold=0, invert=False):
    sobel_compress(file)
    img = Img.open(f"{file[:-4]}_compressed{file[-4:]}")
    img = img.convert('RGB')
    pixis = img.load()
    x, y = img.size
    raw = list()
    for i in range(1, x):
        for j in range(1, y):
            if not invert:
                if pixis[i, j] > (threshold, threshold, threshold):
                    raw.append([i, j, 0])
                    pixis[i, j] = (255, 0, 0)
            else:
                if pixis[i, j] < (threshold, threshold, threshold):
                    raw.append([i, j, 0])
                    pixis[i, j] = (255, 0, 0)

    dotes = np.array(raw, dtype=float).reshape((len(raw), 3))
    print(f"{len(dotes) * 2} dotes, so there are {len(dotes) * 6} faces")
    img.save(f"{file[:-4]}_to_show.png")
    return dotes


def write_result_to_db(res):
    connection = sqlite3.connect('logs.db')
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS 
    Logs (id INTEGER PRIMARY KEY,result TEXT NOT NULL,date TEXT NOT NULL)''')
    date = datetime.date.today()
    last_log = cursor.execute("""SELECT date FROM Logs WHERE id==(SELECT MAX(id) FROM Logs)""").fetchall()

    if str(last_log)[3:-4] != str(date):
        cursor.execute("""DELETE FROM Logs """)

    if res:
        cursor.execute(f"""INSERT INTO Logs (result, date) VALUES ('success', '{date}')""")

    else:
        cursor.execute(f"""INSERT INTO Logs (result, date) VALUES ('fail', {date})""")
    connection.commit()
    connection.close()


def dots_to_3d(file="y.jpg", heigh=1, threshold=0, invert=False):
    data = pict_slc(file, threshold, invert)
    if len(data) == 0:
        return 0
    model = mesh.Mesh(np.zeros(len(data) * 12, dtype=mesh.Mesh.dtype))

    for g, fragment in enumerate(data):
        vertices = np.array([[fragment[0] - 0.5, fragment[1] - 0.5, 0],
                             [fragment[0] - 0.5, fragment[1] + 0.5, 0],
                             [fragment[0] + 0.5, fragment[1] + 0.5, 0],
                             [fragment[0] + 0.5, fragment[1] - 0.5, 0],
                             [fragment[0] - 0.5, fragment[1] + 0.5, heigh],
                             [fragment[0] + 0.5, fragment[1] - 0.5, heigh],
                             [fragment[0] + 0.5, fragment[1] + 0.5, heigh],
                             [fragment[0] - 0.5, fragment[1] - 0.5, heigh],
                             ])
        hull = spatial.ConvexHull(vertices)
        faces = hull.simplices

        for j, f in enumerate(faces):
            for _ in range(3):
                model.vectors[j + g * len(faces) - 1][_] = vertices[f[_], :]
    model.save(f"{file[:-3]}stl")


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("untitled.ui", self)
        self.searchButton.clicked.connect(self.search_browser)
        self.convertButton.clicked.connect(self.convert)
        self.show()
        self.path_source = ""
        self.height_bar.setText("15")
        self.treshold_slider.sliderReleased.connect(self.update_mod_prew)
        self.Invert_checkbox.stateChanged.connect(self.update_mod_prew)

    def update_mod_prew(self):
        if self.path_source == "":
            self.status_label.setText("ERR: File wasn't selected")
            write_result_to_db(0)
            return 0

        pict_slc(self.path_source[0], threshold=int(self.treshold_slider.value()),
                 invert=self.Invert_checkbox.isChecked())

        self.path_out.setText("Preview: \n" + f"{self.path_source[0][:-4]}_to_show.png")
        self.path_modif = f"{self.path_source[0][:-4]}_to_show.png"
        self.preview_modif = QPixmap(self.path_modif).scaled(self.img_modif.size(), QtCore.Qt.KeepAspectRatio)
        self.img_modif.setPixmap(self.preview_modif)

    def convert(self):
        self.height_bar.setText(self.height_bar.text().replace(" ", ""))
        if self.path_source == "":
            self.status_label.setText("ERR: File wasn't selected")
            self.status_label.setStyleSheet("background-color: red")
            write_result_to_db(0)
            return 0
        for i in self.height_bar.text():
            if (i not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]) or self.height_bar.text()[0] == ".":
                self.status_label.setText("ERR: Height data isn't correct")
                self.status_label.setStyleSheet("background-color: red")
                write_result_to_db(0)
                self.height_bar.setText("")
                return 0

        dots_to_3d(f"{self.path_source[0]}", int(self.height_bar.text()),
                   threshold=int(self.treshold_slider.value()), invert=self.Invert_checkbox.isChecked())

        self.status_label.setText("      Done!")
        self.status_label.setStyleSheet("background-color: lightgreen")
        write_result_to_db(1)
        os.remove(f"{self.path_source[0][:-4]}_to_show.png")
        os.remove(f"{self.path_source[0][:-4]}_compressed{self.path_source[0][-4:]}")
        self.path_out.setText("Model: \n" + f'{self.path_source[0][:-3]}stl')

    def search_browser(self):
        self.path_source = QFileDialog.getOpenFileName(self, "Open File", "c:\\",
                                                       "PNG Files (*.png);;JPG Files (*.jpg)")
        if self.path_source == ('', ''):
            return 0
        for i in self.path_source[0]:
            if i in "йцукенгшщзхъфывапролджэёячсмитьбю":
                self.status_label.setText("ERR: Change name of file")
                self.status_label.setStyleSheet("background-color: red")
                return 0

        self.path_in.setText(self.path_source[0])
        self.preview_source = QPixmap(self.path_source[0]).scaled(self.img_source.size(), QtCore.Qt.KeepAspectRatio)
        self.img_source.setPixmap(self.preview_source)

        pict_slc(self.path_source[0], threshold=int(self.treshold_slider.value()),
                 invert=self.Invert_checkbox.isChecked())

        self.path_out.setText("Preview: \n" + f"{self.path_source[0][:-4]}_to_show.png")
        self.path_modif = f"{self.path_source[0][:-4]}_to_show.png"
        self.preview_modif = QPixmap(self.path_modif).scaled(self.img_modif.size(), QtCore.Qt.KeepAspectRatio)
        self.img_modif.setPixmap(self.preview_modif)


app = QApplication(sys.argv)
Window = MainWindow()
app.exec()
