import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from sklearn.utils import shuffle
import numpy as np
import random
import math
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
percptionform1 = Tk()
def drowmat(f1, f2):
    X1 = 0
    X2 = 0
    feature1, feature2, feature3, feature4, Classes =loadFile('IrisData.txt')

    if (f1 == "X1"):
        X1 = 0
    if (f1 == "X2"):
        X1 = 1
    if (f1 == "X3"):
        X1 = 2
    if (f1 == "X4"):
        X1 = 3
    if (f2 == "X1"):
        X2 = 0
    if (f2 == "X2"):
        X2 = 1
    if (f2 == "X3"):
        X2 = 2
    if (f2 == "X4"):
        X2 = 3

    label = []
    for index, value in enumerate(Classes):

        if value == "Iris-setosa":
            label.append(0)
        if value == "Iris-versicolor":
            label.append(1)
        if value == "Iris-virginica":
            label.append(2)
    big_array=[]
    big_array.append(feature1)
    big_array.append(feature2)
    big_array.append(feature3)
    big_array.append(feature4)
    x_min, x_max = big_array[X1].min() ,big_array[X1].max()
    y_min, y_max = big_array[X2].min() , big_array[X2].max()

    plt.figure('fig1')
    plt.clf()
    # Plot the training points
    plt.scatter(big_array[X1], big_array[X2], c=label, cmap=plt.cm.Set1,
                edgecolor='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
   #get data from documnet
def loadFile(df):

    resultList = []
    f = open(df, 'r')
    for line in f:
        line = line.rstrip('\n')
        sVals = line.split(',')
        fVals = list(map(np.str, sVals))
        resultList.append(fVals)
    f.close()
    iris = datasets.load_iris()
    data = np.asarray(iris.data)
    df = pd.DataFrame(data)
    data= np.asarray(resultList, dtype=np.str)
    df1 = pd.DataFrame(data)
    X1 = df[0]
    X2 =  df[1]
    X3 =  df[2]
    X4 = df[3]
    Classes=df1[4]
    return X1,X2,X3,X4,Classes
   #get data i need to train
def getdata_i_need_class(class_one,class_two):
    list1=[]
    list2=[]
    list3=[]
    class_label=[]
    big_list=[]
    feature1, feature2, feature3, feature4, Classes = loadFile('IrisData.txt')
    label1 = np.array( Classes[0:50])
    label2 =np.array(  Classes[50:100])
    label3 =np.array(  Classes[100:151])
    class_label.append(label1)
    class_label.append(label2)
    class_label.append(label3)
    ############################
    feature1_class_one =np.array( feature1[0:50])
    feature1_class_two = np.array( feature1[50:100])
    feature1_class_three = np.array( feature1[100:151])
    list1.append(feature1_class_one)
    list2.append(feature1_class_two)
    list3.append(feature1_class_three)
    ###########################
    feature2_class_one =np.array(  feature2[0:50])
    feature2_class_two =np.array(  feature2[50:100])
    feature2_class_three =np.array(  feature2[100:151])
    list1.append(feature2_class_one)
    list2.append(feature2_class_two)
    list3.append(feature2_class_three)
    #########################
    feature3_class_one =np.array(  feature3[0:50])
    feature3_class_two =np.array(  feature3[50:100])
    feature3_class_three =np.array(  feature3[100:151])
    list1.append(feature3_class_one)
    list2.append(feature3_class_two)
    list3.append(feature3_class_three)
    #############################
    feature4_class_one =np.array(  feature4[0:50])
    feature4_class_two =np.array(  feature4[50:100])
    feature4_class_three =np.array(  feature4[100:151])
    list1.append(feature4_class_one)
    list2.append(feature4_class_two)
    list3.append(feature4_class_three)
    big_list.append(list1)
    big_list.append(list2)
    big_list.append(list3)
    big_list.append(class_label)
    return big_list[class_one],big_list[class_two],big_list
def get_fetuer(f1,f2,class_one,class_two):
        data1,data2,big=getdata_i_need_class(class_one,class_two)
        fetuer_one= make_one_list(data1[f1],data2[f1])
        fetuer_two= make_one_list(data1[f2],data2[f2])
        classes=make_one_list(big[3][class_one],big[3][class_two])
        label=[]
        classes.sort()
        d = 1
        for index,value  in enumerate (classes):
           count1=0
           count2=0
           count3=0

           if value=="Iris-setosa" :
               count1 += 1
               if count1>=0:
                 label.append(-1)
                 d = 0
           if value=="Iris-versicolor" :
               count2 += 1
               if count2 >= 0:
                 if d>0:
                     label.append(-1)
                 else:
                     label.append(1)
           if value=="Iris-virginica" :
               count3 += 1
               if count1 >= 0:
                 label.append(1)
                 d+=1

        df=pd.DataFrame(
        {'f1': fetuer_one,
         'f2': fetuer_two,
         'label': label
        })

        return df
   # print(data2)
def make_one_list(list1,list2):
    new_list=[]
    for index,value in  enumerate(list1):
          new_list.append(value)
    for index, value in enumerate(list2):
        new_list.append(value)
    return new_list


  #signum
def signum_function(result):
    if (result == 0):
         result_label = 1
    elif(result>0):
         result_label=1
    else:
         result_label=-1
    return result_label
def creat_wight():
     Weight_Matrix = np.empty([3, 1], dtype=float)
     weight = np.random.rand(3, 1)
     for i in range(0, 3):
        Weight_Matrix[i] = weight[i]
     return Weight_Matrix
def Layer_pers(datafram, Epochs, Learing_rat, bais):
    # Epochs=10
    # LR=10
    listbais=[bais]*len(datafram['f1'])
    listbais1=np.array( listbais)
    lab=np.array(datafram['label'])
    listbais1.reshape(1,60)
    lif1= np.array( datafram['f1'])
    lif2 = np.array(datafram['f2'])
    lif1.reshape(1,60)
    lif2.reshape(1, 60)
    Weight = creat_wight()
    for E in range(0, Epochs):
        print("epouc:")
        for i in range(0, 60):
            Recourd_input = np.empty([1, 3])
            Recourd_input[0][0] = listbais[i]
            Recourd_input[0][1] =lif1[i]
            Recourd_input[0][2] = lif2[i]
            #𝑦_𝑖 = signum 〖[𝑤(𝑖)〗^𝑇. 𝑥(𝑖)+𝑏]
            NormaizedVal=signum_function(np.dot(Recourd_input, Weight))
            if (NormaizedVal !=lab[i]):
                #if 𝑦_𝑖 ≠ 𝑡_𝑖  then
                #calculate loss L = (𝑡_𝑖  - 𝑦_𝑖)
                Loss=lab[i]-NormaizedVal
                Term=np.dot(Learing_rat,Loss)
                # form a new weight vector wi+1 according to
                #𝑤_(𝑖+1) = 𝑤_𝑖 + η.(𝑡_𝑖  - 𝑦_𝑖).𝑥_𝑖
                Weight[0][0] = Weight[0][0]+(np.dot(Term,Recourd_input[0][0]))
                Weight[1][0] = Weight[1][0]+(np.dot(Term,Recourd_input[0][1]))
                Weight[2][0] = Weight[2][0]+(np.dot(Term,Recourd_input[0][2]))

    return Weight



def Adline(datafram,Learing_rat, bais,mse):
    listbais = [bais] * len(datafram['f1'])
    listbais1 = np.array(listbais)
    lab = np.array(datafram['label'])
    listbais1.reshape(1, 60)
    lif1 = np.array(datafram['f1'])
    lif2 = np.array(datafram['f2'])
    lif1.reshape(1, 60)
    lif2.reshape(1, 60)
    epoch=0
    loss2 = 0
    x=999999999
    Weight = creat_wight()
    while(x>mse):
        for index in range(60):
            new_matrex_input1=np.empty([1,3])
            new_matrex_input1[0][0]=listbais1[index]
            new_matrex_input1[0][1]=lif1[index]
            new_matrex_input1[0][2]=lif2[index]
            y=np.dot(new_matrex_input1,Weight)
            e=lab[index]-y
            t=np.dot(Learing_rat,e)
            Weight[0][0] = Weight[0][0] + (np.dot(t, new_matrex_input1[0][0]))
            Weight[1][0] = Weight[1][0] + (np.dot(t, new_matrex_input1[0][1]))
            Weight[2][0] = Weight[2][0] + (np.dot(t, new_matrex_input1[0][2]))
        for i in range(0, 60):
            new_matrex_input2 = np.empty([1, 3])
            new_matrex_input2[0][0] = listbais1[i]
            new_matrex_input2[0][1] = lif1[i]
            new_matrex_input2[0][2] = lif2[i]
            y2 = np.dot(new_matrex_input2, Weight)
            sig=signum_function(y2)
            if sig !=lab[i]:
                #loss=((error1^2)+(error2^2))/2
               loss2 += np.nan_to_num(((lab[i] - y2) * (lab[i] - y2) / 2))
        #print(loss2)
        x = (loss2 / 60)
        epoch += 1
        if epoch==100:
            break
    print(x)
    print("Epochs is: ", epoch)
    return Weight


def draw_line(F1_input,F2_input,LabelList_input,Updatedweight):
    LabelList =np.array(LabelList_input)
    F1=np.array(F1_input)
    F2=np.array(F2_input)
    f1_c1=[]
    f1_c2=[]
    f2_c1=[]
    f2_c2=[]
    for i in range(0,40):
        if(LabelList[i]==1):
            f1_c1.append(F1[i])
            f2_c1.append(F2[i])
        elif(LabelList[i]==-1):
            f1_c2.append(F1[i])
            f2_c2.append(F2[i])
    plt.figure('fig1')
    plt.scatter(f1_c1,f2_c1)
    plt.scatter(f1_c2,f2_c2)
    minx=min(F1)
    maxx=max(F1)
    maxy = (-(Updatedweight[1] * maxx) - Updatedweight[0]) / Updatedweight[2]
    miny=(-(Updatedweight[1]*minx)-Updatedweight[0])/Updatedweight[2]
    plt.plot((minx,maxx),(miny,maxy))
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.show()
def test(test_set, wight):
    confusionmatrix = np.empty([2, 2])
    listbais = [1] * len(test_set['f1'])
    listbais1 = np.array(listbais)
    lab = np.array(test_set['label'])
    listbais1.reshape(1, 40)
    lif1 = np.array(test_set['f1'])
    lif2 = np.array(test_set['f2'])
    lif1.reshape(1, 40)
    lif2.reshape(1, 40)
    counter = 0
    Accuracy = 0
    class_one_True = 0
    class_two_tru = 0
    class_one_false = 0
    class_two_false = 0
    for index, va in enumerate(lab):
        InputRecordMat = np.empty([1, 3])
        InputRecordMat[0][0] = listbais[index]
        InputRecordMat[0][1] = lif1[index]
        InputRecordMat[0][2] = lif2[index]
        NetMatrix = np.dot(InputRecordMat, wight)
        NormaizedVal = signum_function(NetMatrix)
        if (lab[index] == NormaizedVal and lab[index]== 1):
            class_one_True += 1
            Accuracy += 1
        elif (lab[index] == NormaizedVal and lab[index] == -1):
            class_two_tru += 1
            Accuracy += 1
        elif (lab[index] != NormaizedVal and lab[index] == -1):
            class_two_false += 1
        elif (lab[index] != NormaizedVal and lab[index]== 1):
            class_one_false += 1

        confusionmatrix[0][0] = class_one_True
        confusionmatrix[1][0] = class_two_tru
        confusionmatrix[0][1] = class_one_false
        confusionmatrix[1][1] =  class_two_false
        if (NormaizedVal == lab[index]):
            counter += 1
    return ((counter / 40) * 100),confusionmatrix
def run(fetuer1_, fetuer2_, class1_, class2_, bais, Epoch):

        X1 =0
        X2 =0
        Class1 =0
        Class2 =0
        if(class1_=="setosa"):
            Class1=0
        if (class1_ == "versicolor"):
            Class1 = 1
        if (class1_ == "virginica"):
            Class1 = 2
        if (class2_ == "setosa"):
            Class2 = 0
        if (class2_ == "versicolor"):
            Class2 = 1
        if (class2_ == "virginica"):
            Class2 = 2
        if(fetuer1_=="X1"):
            X1=0
        if(fetuer1_=="X2"):
            X1=1
        if (fetuer1_ == "X3"):
            X1 = 2
        if (fetuer1_ == "X4"):
            X1 = 3
        if (fetuer2_ == "X1"):
            X2 = 0
        if (fetuer2_ == "X2"):
            X2 = 1
        if (fetuer2_ == "X3"):
            X2 = 2
        if (fetuer2_ == "X4"):
            X2 = 3
        datafram = get_fetuer(X1, X2, Class1, Class2)
        datafram_copy = datafram.copy()
        datafram_copy = shuffle(datafram_copy)
        train_set = datafram_copy.sample(frac=0.60, random_state=1)
        test_set = datafram_copy.drop(train_set.index)
        wight = Layer_pers(train_set,int( Epoch), 0.05, int(bais))
        wight2= Adline(train_set,0.005,int(bais),0.05)
        acuuracy, confution_matrix =test(test_set, wight)
        acuuracy2, confution_matrix2 = test(test_set, wight2)
        print('Task one','\n','Accuracy=', int(acuuracy), '%', '\n', 'Confusion Matrix is:\n', confution_matrix, '\n')
        print('Task two','\n','Accuracy2=', int(acuuracy2), '%', '\n', 'Confusion Matrix2 is:\n', confution_matrix2, '\n')

        draw_line(test_set['f1'], test_set['f2'], test_set['label'], wight)
        draw_line(test_set['f1'], test_set['f2'], test_set['label'], wight2)
def Validation(feature11, feature22, class11, class22, learning_rate, number_epoch1):
    fetuer1 = str(feature11)
    fetuer2 = str(feature22)
    class1 = str(class11)
    class2 = str(class22)
    if (fetuer1 == "" or fetuer2 == "" or class1 == ""
            or class2 == "" or learning_rate == "" or number_epoch1 == ""
           ):
        msg = messagebox.showinfo("Please Enter Data ")
        return 0
    return 1

def form():

    percptionform1.geometry("700x200")
    # lables
    lab_feature1 = Label(percptionform1, text="Feature1")
    lab_feature1.grid(row=1, column=0, padx=1, pady=1, ipady=1, ipadx=1)

    lab_feature2 = Label(percptionform1, text="class1")
    lab_feature2.grid(row=2, column=0, padx=1, pady=1, ipady=1, ipadx=1)

    lab_class1 = Label(percptionform1, text="Feature2")
    lab_class1.grid(row=1, column=2, padx=1, pady=1, ipady=1, ipadx=1)

    lab_class2 = Label(percptionform1, text="class2")
    lab_class2.grid(row=2, column=2, padx=1, pady=1, ipady=1, ipadx=1)

    lab_learning = Label(percptionform1, text="learning rate")
    lab_learning.grid(row=3, column=0, padx=1, pady=1, ipady=1, ipadx=1)

    lab_epochs = Label(percptionform1, text="number of epochs")
    lab_epochs.grid(row=3, column=2, padx=1, pady=1, ipady=1, ipadx=1)

    lab_bias = Label(percptionform1, text="bias")
    lab_bias.grid(row=3, column=4, padx=1, pady=1, ipady=1, ipadx=1)

    # ComboBoxes
    cmb_feature1 = ttk.Combobox(percptionform1, values=["X1", "X2", "X3", "X4"])
    cmb_feature1.grid(row=1, column=1, padx=4, pady=4, ipady=2, ipadx=2)

    cmb_class1 = ttk.Combobox(percptionform1, values=["setosa", "versicolor", "virginica"])
    cmb_class1.grid(row=2, column=1, padx=4, pady=4, ipady=2, ipadx=2)

    cmb_feature2 = ttk.Combobox(percptionform1, values=["X1", "X2", "X3", "X4"])
    cmb_feature2.grid(row=1, column=3, padx=4, pady=4, ipady=2, ipadx=2)

    cmb_class2 = ttk.Combobox(percptionform1, values=["setosa", "versicolor", "virginica"])
    cmb_class2.grid(row=2, column=3, padx=4, pady=4, ipady=2, ipadx=2)

    # Text
    learning_rate_txt = Entry(percptionform1)
    learning_rate_txt.grid(row=3, column=1, padx=5, pady=5, ipady=2, ipadx=3)

    epochs_txt = Entry(percptionform1)
    epochs_txt.grid(row=3, column=3, padx=5, pady=5, ipady=2, ipadx=3)

    bias_txt = Entry(percptionform1)
    bias_txt.grid(row=3, column=5, padx=5, pady=5, ipady=2, ipadx=3)


    # Button
    run_button = Button(percptionform1, text="Run task 1/2",
                        command=lambda:run(cmb_feature1.get(),cmb_feature2.get(),cmb_class1.get(),cmb_class2.get(),bias_txt.get(),epochs_txt.get()))
    run_button.grid(row=5, column=1, padx=3, pady=3, ipady=10, ipadx=10)
    Draw_button = Button(percptionform1, text="Draw" ,command=lambda:drowmat(cmb_feature1.get(),cmb_feature2.get()))
    Draw_button.grid(row=5, column=2, padx=3, pady=3, ipady=10, ipadx=10)

    percptionform1.mainloop()

form()


# run()