import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import seaborn as sns

from modules import cleaning
from modules import main_func
from modules import graphs


#creates main window
class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent=None)
        vLayout = QtWidgets.QVBoxLayout(self)
        hLayout = QtWidgets.QHBoxLayout()

        #Import data
        self.loadBtn = QtWidgets.QPushButton("Select File", self)
        hLayout.addWidget(self.loadBtn)
        vLayout.addLayout(hLayout)

        #table that displays the data
        self.pandasTv = QtWidgets.QTableView(self)
        vLayout.addWidget(self.pandasTv)
        self.loadBtn.clicked.connect(self.loadFile)

        #Present insights and summaries from data analysis
        #first graph
        self.displayBtn = QtWidgets.QPushButton("Display 1st", self)
        hLayout.addWidget(self.displayBtn)
        self.displayBtn.clicked.connect(self.displayGraph1)

        #second graph
        self.displayBtn2 = QtWidgets.QPushButton("Display 2nd", self)
        hLayout.addWidget(self.displayBtn2)
        self.displayBtn2.clicked.connect(self.displayGraph2)


        # Using a heatmap to show if there is a correlation or if there isn't
        # Y-Axis: Do you feel employers take mental health as seriously as physical health
        # X-Axis: Do you believe your productivity is ever affected by a mental health issue?
        self.displayBtn3 = QtWidgets.QPushButton("Display 3rd", self)
        hLayout.addWidget(self.displayBtn3)
        self.displayBtn3.clicked.connect(self.displayGraph3)


        # age group
        self.displayBtn4 = QtWidgets.QPushButton("Display 4th", self)
        hLayout.addWidget(self.displayBtn4)
        self.displayBtn4.clicked.connect(self.displayGraph4)



        # table that displays the data
        # self.barGraph = QtWidgets.QTableView(self)
        # vLayout.addWidget(self.barGraph)
        # self.displayBtn.clicked.connect(self.displayGraphs)



    def loadFile(self):
        df = main_func.data_import()
        cleaning.clean(df)
        model = main_func.PandasModel(df)
        self.pandasTv.setModel(model)


    def displayGraph1(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        graph = graphs.workPer(df)

    def displayGraph2(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        g = graphs.conditions(df)
        #self.barGraph.setModel(graph)


    def displayGraph3(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        graphs.heatmap(df)

    def displayGraph4(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        print('ok')         #cleaning no issue
        graphs.ageGroup(df) #issue in function
        print('ok2')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())





