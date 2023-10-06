

import pandas as pd
import numpy as np

import subprocess

import warnings
warnings.filterwarnings('ignore')

from tkinter import filedialog

import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize

from modules import cleaning
from modules import main_func

# filepath = filedialog.askopenfilename(title="Open a Text File", filetypes=(("all files","*.*"), ("text    files","*.txt")))
# file = open(filepath, encoding="utf8")
#
# df = pd.read_csv(file)

#creates main window
class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setGeometry(300, 300, 1280, 780)
        self.setWindowTitle("Mental Health Data Analysis")

        pybutton = QPushButton('Choose file', self)
        pybutton.clicked.connect(self.clickMethod)
        pybutton.resize(100,32)
        pybutton.move(50, 50)


    #import function
    def clickMethod(self):
        df = main_func.data_import()
        cleaning.clean(df)
        print(df)




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit( app.exec_() )





#print(df)





