
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import *

from modules import cleaning
from modules import main_func
from modules import graphs


#creates main window
class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent=None)
        self.setWindowTitle('INF-1002 Mental Health System')
        self.setMinimumSize(1000,800)
        vLayout = QtWidgets.QVBoxLayout(self)
        hLayout = QtWidgets.QHBoxLayout()


        #Import data
        headerInsight = QtWidgets.QLabel("Import file to display:")
        self.loadBtn = QtWidgets.QPushButton("Select File", self)
        self.loadBtn.setFixedSize(80, 25)
        hLayout.setContentsMargins(0, 0, 750, 0)
        hLayout.addWidget(headerInsight)
        hLayout.addWidget(self.loadBtn)
        vLayout.addLayout(hLayout)

        #table that displays the data
        self.pandasTv = QtWidgets.QTableView(self)
        vLayout.addWidget(self.pandasTv)
        self.loadBtn.clicked.connect(self.loadFile)

        headerInsight = QtWidgets.QLabel("Our Insights:")
        headerInsight.setFont(QFont('Arial', 18))
        vLayout.addWidget(headerInsight)

        #### Present insights and summaries from data analysis section ####
        # Section to display graphs
        mainGrid = QtWidgets.QGridLayout()
        mainGrid.setColumnMinimumWidth(0, 100)

        # first grid sec
        firstGrid = QtWidgets.QGridLayout()
        firstGrid.setContentsMargins(0, 5, 10, 0)
        firstGrid.setColumnMinimumWidth(1, 100)

        # first grid sec - right
        firstGridR = QtWidgets.QGridLayout()
        firstGridR.setContentsMargins(0, 0, 10, 0)

        secGrid = QtWidgets.QGridLayout()
        secGrid.setContentsMargins(0, 15, 10, 0)
        # secGrid.setRowStretch(3, 1)

        vLayout.addLayout(mainGrid)
        mainGrid.addLayout(firstGrid, 0, 0)
        mainGrid.addLayout(firstGridR, 0, 1)
        mainGrid.addLayout(secGrid, 1, 0)





        #Use firstGrid
        # How does mental health affect work performance?
        peformanceLabel = QtWidgets.QLabel("How does mental health affect work performance?:")
        self.displayBtn = QtWidgets.QPushButton("Display", self)
        self.displayBtn.setFixedSize(80, 25)
        self.displayBtn.clicked.connect(self.displayGraph1)
        firstGrid.addWidget(peformanceLabel, 0, 0)
        firstGrid.addWidget(self.displayBtn, 0, 1)

        # What are the common mental health combinations?
        conditionsLabel = QtWidgets.QLabel("What are the common mental health combinations?:")
        self.displayBtn2 = QtWidgets.QPushButton("Display", self)
        self.displayBtn2.setFixedSize(80, 25)
        self.displayBtn2.clicked.connect(self.displayGraph2)
        firstGrid.addWidget(conditionsLabel, 1, 0)
        firstGrid.addWidget(self.displayBtn2, 1, 1)

        #Work time affected by health
        workAff = QtWidgets.QLabel("Work time affected by heath:")
        self.displayBtn3 = QtWidgets.QPushButton("Display", self)
        self.displayBtn3.setFixedSize(80, 25)
        self.displayBtn3.clicked.connect(self.displayGraph3)
        firstGrid.addWidget(workAff, 2, 0)
        firstGrid.addWidget(self.displayBtn3, 2, 1)


        # Using a heatmap to show if there is a correlation or if there isn't
        # X-Axis: Do you believe your productivity is ever affected by a mental health issue?
        # Y-Axis: Do you feel employers take mental health as seriously as physical health?
        heatmapLabel = QtWidgets.QLabel("Correlation between Employer taking \nmental health serious vs productivity")
        self.displaybtnHeat = QtWidgets.QPushButton("Display", self)
        self.displaybtnHeat.setFixedSize(80, 25)
        self.displaybtnHeat.clicked.connect(self.displayHeat)
        firstGrid.addWidget(heatmapLabel, 3, 0)
        firstGrid.addWidget(self.displaybtnHeat, 3, 1)






        #uses firstGridR


        # #heatmap correlation
        # corrHeat = QtWidgets.QLabel("Overall Correlation:")
        # self.corrBtn = QtWidgets.QPushButton("Display", self)
        # self.corrBtn.setFixedSize(80, 25)
        # self.corrBtn.clicked.connect(self.overallHeat)
        # firstGridR.addWidget(corrHeat, 1, 0)
        # firstGridR.addWidget(self.corrBtn, 1, 1)


        # Are current efforts effective and enough?
        effortLabel = QtWidgets.QLabel("Are current efforts effective and enough?")
        effortLabel.setContentsMargins(0, 20, 0, 0)
        self.dropdownEff = QtWidgets.QComboBox()
        self.dropdownEff.setFixedSize(300, 25)
        self.dropdownEff.addItems(["Mental health benefits in healthcare coverage", "Discussed or Conducted mental health events",
                                   "Provide extra mental health resource"])
        firstGridR.addWidget(effortLabel, 2, 0)
        firstGridR.addWidget(self.dropdownEff, 3, 0)

        # Efforts
        self.displayBtn8 = QtWidgets.QPushButton("Display", self)
        self.displayBtn8.setFixedSize(80, 25)
        self.displayBtn8.clicked.connect(self.displayGraph8)
        firstGridR.addWidget(self.displayBtn8, 3, 1)



        # Consequence upon discussing Mental Health Disorders with Employers
        conseqLabel = QtWidgets.QLabel("Consequence upon discussing Mental Health Disorders \nwith Employers:")
        conseqLabel.setContentsMargins(0, 20, 0, 0)
        self.dropdownCon = QtWidgets.QComboBox()
        self.dropdownCon.setFixedSize(300, 25)
        self.dropdownCon.addItems(["Employees with No Mental Health Support", "Employees with All Mental Health Support"])
        firstGridR.addWidget(conseqLabel, 4, 0)
        firstGridR.addWidget(self.dropdownCon, 5, 0)

        # Consequences
        self.displayConBtn = QtWidgets.QPushButton("Display", self)
        self.displayConBtn.setFixedSize(80, 25)
        self.displayConBtn.clicked.connect(self.displayConsq)
        firstGridR.addWidget(self.displayConBtn, 5, 1)



        # Likeliness to Discuss Mental Health, Influenced by Observations/Experiences
        likeLabel = QtWidgets.QLabel("Likeliness to Discuss Mental Health, Influenced \nby Observations/Experiences:")
        likeLabel.setContentsMargins(0, 20, 0, 0)
        self.displayLikeBtn = QtWidgets.QPushButton("Display", self)
        self.displayLikeBtn.setFixedSize(80, 25)
        self.displayLikeBtn.setContentsMargins(0, 25, 0, 0)
        self.displayLikeBtn.clicked.connect(self.infu)
        firstGridR.addWidget(likeLabel, 6, 0)
        firstGridR.addWidget(self.displayLikeBtn, 6, 1)



        # Binary Logic Regression
        binLabel = QtWidgets.QLabel("Prediction of whether certain factors \ndetermine mental illness:")
        binLabel.setContentsMargins(0, 20, 0, 0)
        self.displayBtn9 = QtWidgets.QPushButton("Display", self)
        self.displayBtn9.setFixedSize(80, 25)
        self.displayBtn9.setContentsMargins(0, 25, 0, 0)
        self.displayBtn9.clicked.connect(self.displayBin)
        firstGridR.addWidget(binLabel, 7, 0)
        firstGridR.addWidget(self.displayBtn9, 7, 1)






        # secGrid
        # Prone to mental health issue
        groups = QtWidgets.QLabel("What groups are more prone to mental health issues?")
        secGrid.addWidget(groups, 0, 0)

        # AGE INSIGHT - Prone to mental health issues
        age = QtWidgets.QLabel("a) Age")
        secGrid.addWidget(age, 1, 0)

        # dropdown for selected graph for age group
        self.dropdownAge = QtWidgets.QComboBox()
        self.dropdownAge.setFixedSize(150, 25)
        self.dropdownAge.addItems(["Bar graph", "Pie chart"])
        secGrid.addWidget(self.dropdownAge, 1, 1)

        # age group
        self.displayBtn4 = QtWidgets.QPushButton("Display", self)
        self.displayBtn4.setFixedSize(80, 25)
        self.displayBtn4.clicked.connect(self.displayGraph4)
        secGrid.addWidget(self.displayBtn4, 1, 2)



        # GENDER INSIGHT - Prone to mental health issues
        gender = QtWidgets.QLabel("b) Gender")
        secGrid.addWidget(gender, 2, 0)

        # dropdown for selected graph for gender
        self.dropdownGen = QtWidgets.QComboBox()
        self.dropdownGen.setFixedSize(150, 25)
        self.dropdownGen.addItems(["Bar graph", "Pie chart"])
        secGrid.addWidget(self.dropdownGen, 2, 1)

        # gender group
        self.displayBtn5 = QtWidgets.QPushButton("Display", self)
        self.displayBtn5.setFixedSize(80, 25)
        self.displayBtn5.clicked.connect(self.displayGraph5)
        secGrid.addWidget(self.displayBtn5, 2, 2)



        # FAMILY HISTORY INSIGHT - Prone to mental health issues
        family = QtWidgets.QLabel("c) Family history of mental illness")
        secGrid.addWidget(family, 3, 0)

        # dropdown for selected graph for family history
        self.dropdownFam = QtWidgets.QComboBox()
        self.dropdownFam.setFixedSize(150, 25)
        self.dropdownFam.addItems(["Bar graph", "Pie chart"])
        secGrid.addWidget(self.dropdownFam, 3, 1)

        # family history
        self.displayBtn6 = QtWidgets.QPushButton("Display", self)
        self.displayBtn6.setFixedSize(80, 25)
        self.displayBtn6.clicked.connect(self.displayGraph6)
        secGrid.addWidget(self.displayBtn6, 3, 2)



        # LOCATION INSIGHT - Prone to mental health issues
        location = QtWidgets.QLabel("d) Location")
        secGrid.addWidget(location, 4, 0)

        # dropdown for selected graph for location
        self.dropdownLoc = QtWidgets.QComboBox()
        self.dropdownLoc.setFixedSize(150, 25)
        self.dropdownLoc.addItems(["Bar graph", "Pie chart"])
        secGrid.addWidget(self.dropdownLoc, 4, 1)

        # location
        self.displayBtn7 = QtWidgets.QPushButton("Display", self)
        self.displayBtn7.setFixedSize(80, 25)
        self.displayBtn7.clicked.connect(self.displayGraph7)
        secGrid.addWidget(self.displayBtn7, 4, 2)



        # # classification report (MIGHT REMOVE)
        # classify = QtWidgets.QLabel("Overall classification:")
        # secGrid.addWidget(classify, 5, 0)
        #
        # # classify
        # self.overallBtn = QtWidgets.QPushButton("Display", self)
        # self.overallBtn.setFixedSize(80, 25)
        # self.overallBtn.clicked.connect(self.classfiedGraph)
        # secGrid.addWidget(self.overallBtn, 5, 1)





    ### functions to display graphs ###
    def loadFile(self):
        df = main_func.data_import()
        cleaning.clean(df)
        model = main_func.PandasModel(df)
        self.pandasTv.setModel(model)


    def displayGraph1(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        graphs.workPer(df)

    def displayGraph2(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        graphs.conditions(df)


    def displayGraph3(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        graphs.workAffected(df)

    def displayHeat(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        graphs.heatmap(df)

    def displayGraph4(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        content = self.dropdownAge.currentText()
        graphs.ageGroup(df, content)

    def displayGraph5(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        content = self.dropdownGen.currentText()
        graphs.gender(df, content)


    def displayGraph6(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        content = self.dropdownFam.currentText()
        graphs.famHistory(df, content)


    def displayGraph7(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        content = self.dropdownLoc.currentText()
        graphs.location(df, content)

    def displayGraph8(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        content = self.dropdownEff.currentText()
        graphs.effort(df, content)

    def displayConsq(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        content = self.dropdownCon.currentText()
        graphs.conseq(df, content)


    def infu(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        graphs.likeInf(df)

    def displayBin(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        graphs.binaryLog(df)


    # ############(GOT ISSUES)
    # def classfiedGraph(self):
    #     df = pd.read_csv('data/mental-heath.csv')
    #     cleaning.clean(df)
    #     graphs.classfied(df)
    # ######################

    def overallHeat(self):
        df = pd.read_csv('data/mental-heath.csv')
        cleaning.clean(df)
        graphs.correlationHeat(df)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())





