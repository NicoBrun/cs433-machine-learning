import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import load_csv_data,predict_labels
from matplotlib.widgets import Button, RadioButtons


data_path = "train.csv"
y_binary,input_data,ids = load_csv_data(data_path)
feature1 = 0
feature2 = 0
removeNan = True
blueInFront = True

fig, ax = plt.subplots(figsize=(20,10))
plt.subplots_adjust(left=0.3)


axcolor = 'lightgoldenrodyellow'
rax1 = plt.axes([0.05, 0.1, 0.05, 0.8], facecolor=axcolor)
rax1.set_xlabel('feature 1')
radio1 = RadioButtons(rax1, range(30))
for circle in radio1.circles: # adjust radius here. The default is 0.05
    circle.set_radius(0.02)
rax2 = plt.axes([0.12, 0.1, 0.05, 0.8], facecolor=axcolor)
rax2.set_xlabel('feature 2')
radio2 = RadioButtons(rax2, range(30))
for circle in radio2.circles: # adjust radius here. The default is 0.05
    circle.set_radius(0.02)

rax3 = plt.axes([0.2, 0.1, 0.07, 0.1], facecolor=axcolor)
bColor = Button(rax3, 'Switch top color')


def updateFeature1(feature):
    global feature1
    feature1 = int(feature)
    refreshGraph()

def updateFeature2(feature):
    global feature2
    feature2 = int(feature)
    refreshGraph()

def changeColor(x):
    global blueInFront
    blueInFront = not blueInFront
    refreshGraph()

def refreshGraph():
    global feature1
    global feature2
    global blueInFront
    ax.clear()
    positivElemX = []
    positivElemY = []
    negativElemX = []
    negativElemY = []
    sizeRed = 0
    sizeBlue = 0
    for i, elem in enumerate(y_binary):
        value1 = input_data[i][feature1]
        value2 = input_data[i][feature2]
        if not (removeNan and (value1 == -999 or value2 == -999)):
            if elem == 1:
                positivElemX.append(value1)
                positivElemY.append(value2)
                sizeRed += 1
            else:
                negativElemX.append(value1)
                negativElemY.append(value2)
                sizeBlue += 1
    #red is 1 and blue is 0
    if blueInFront:
        ax.plot(positivElemX, positivElemY, 'ro',negativElemX, negativElemY, 'bs',ms = 1)  
    else:
        ax.plot(negativElemX, negativElemY, 'bs',positivElemX, positivElemY, 'ro',ms = 1) 
    ax.set_xlabel('feature 1')
    ax.set_ylabel('feature 2')
    print('sizeRed:',sizeRed,'sizeBlue',sizeBlue)


refreshGraph()
radio1.on_clicked(updateFeature1)
radio2.on_clicked(updateFeature2)
bColor.on_clicked(changeColor)
plt.show()

#0-1 en rouge par dessus
#0-2 en rouge
#0-11 rouge
#0-20 rouge : on voit que le 20 n'influence pas le rouge
#0 summary: le 0 rouge est surtout entre 67 et 180
#1 summary: le 1 rouge surtout entre 0 et 120
#2 summary: ressemble au 0
#4-5 ressemble à une courbe x^2
#4-6 ressemble à une courbe -x^2
#4-11 il y a une legere ligne bleue sans rouge en y=0
#4-22 le classique feature 22
#5-22 exist if 22 == 2,3
#5 summary: idk, petite ligne en x = 83, par exemple #5-25
#6-12 ??
#6-14 ??
#6 summary: pas mal de rouge en x<0, et une ligne de rouge en x = 0.0129
#6-22, existe seulement si 22 est 2,3 thanks Jojo
#6-24 ??? sablier
#6-27 sablier
#7 summary assez distinct, zone de bleu sans rouge, useful?
#7-9 x^2
#7-11 zone vraiment sans rouge
#7-14 very sexy ???
#8-13 un seul point en 2500, alors que le reste est en <500 (à enlever?)
#8-19 outliner pour les deux features
#9-29 looks linear
#10-13 looks like an inverse function
#10-16 linear
#11-12 tout dans les bordure
#12-22 only if y =2 or 3
#14-17 grande zones sans rouge
#15-17 3 lignes blanches
#15-18 2 lignes + 1 blanches
#16-19 2 outliners points
#17-25 3 lignes blanches
#18-20 est assez dank linges everywhere
#22 we know
#23-26 triangle
#23-29 linear
#24-27 seems strange
#25-28 blue lines (without red)
#26-29 looks like 23-29
#11-12
