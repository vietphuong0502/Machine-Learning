import numpy as np
import pandas as pd 
import tkinter as tk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

window = tk.Tk()
window.title("")
window.geometry("700x600")
name = tk.Label(window, text = "DỰ ĐOÁN TIỀN GIẤY THÔNG QUA CÁC CHỈ SỐ").place(x = 230, y = 10)

Variance_entry = tk.DoubleVar()
Skewness_entry = tk.DoubleVar()
Curtosis_entry = tk.DoubleVar()
Entropy_entry = tk.DoubleVar()


def submit():
    df = pd.read_csv('./banknote_authentication.csv')
    X = np.array(df.loc[:, df.columns != "class"].values)
    y = np.array([df["class"]]).T
    # #chia dữ liệu thành 70% train 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
 	
    # create linear regression object
    model = LinearRegression()
    main_model = LinearRegression()

	# chia du lieu thanh 3 folds 
    kf = KFold(n_splits=3)
    maxScore = 0
    
    # generate indices to split data into training and test set.
    for train_index, test_index in kf.split(X_train, y_train):
        new_X_train, new_X_test = X_train[train_index], X_train[test_index]
        new_y_train, new_y_test = y_train[train_index], y_train[test_index]
        model.fit(new_X_train, new_y_train)
        # score method return the coefficient of determination of the prediction
        score_trained = model.score(new_X_test, new_y_test)
        if score_trained > maxScore:
            maxScore = score_trained
            main_model.fit(new_X_train, new_y_train)  # Truyen du lieu new_X_train, new_y_train cho doi tuong
    
    prediction = main_model.predict(X_test)
    count1 = 0
    #ham for dung de dem so lan y_test = y_dudoan de du doan ty le dung sai
    for i in range(len(prediction)):
        if y_test[i][0] == int(prediction[i][0]):
            count1 +=1
    print("CROSS VALIDATION")
    print('Ty le du doan dung:', count1/len(prediction)*100, "%")
    print('Ty le du doan sai:',100 - count1/len(prediction)*100, "%")
    
    #perceptron dua tren tap du lieu moi
    X_train1, X_test1, y_train1, y_test1 = train_test_split(new_X_train, new_y_train, test_size=0.3, shuffle=False)
    pla = Perceptron()                         # lấy thuật toán perceptron
    pla.fit(X_train1, y_train1)          # đưa ra mô hình tốt nhất của perceptron với dữ liệu đầu vào (X_train,y_train)
    y_pred1 = pla.predict(X_test1)              # đưa ra dự đoán của perceptron
    # du doan cho du lieu nhap
    y_pred = pla.predict([[Variance_entry.get(),Skewness_entry.get(),Curtosis_entry.get(),Entropy_entry.get()]])
    
    count2 = 0
    #ham for dung de dem so lan y_test = y_dudoan de du doan ty le dung sai
    for i in range(0,len(y_test1)):
        if(y_test1[i] == y_pred1[i]): 
            count2 = count2 + 1
    print("PERCEPTRON") 
    print('Ty le du doan dung:', count2/len(y_pred1)*100, "%")    
    print('Ty le du doan sai:',100 - count2/len(y_pred1)*100, "%")
    #đánh giá mo hinh
    acc1 = accuracy_score(y_test1,y_pred1) # accuracy: do chinh xac cua mo hinh 
    pre1 = precision_score(y_test1,y_pred1) # tỉ lệ số điểm true positive trong số những điểm được phân loại là positive (TP + FP).
    rec1 = recall_score(y_test1,y_pred1) # tỉ lệ số điểm true positive trong số những điểm thực sự là positive (TP + FN).
    f1_s1  = f1_score(y_test1,y_pred1) # kết hợp cả Recall và Precision 
    print("accuracy =",acc1)
    print("precision =",pre1)
    print("recall =",rec1)
    print("F1 =", f1_s1)
    
    Accuracy_label_answer["text"]=acc1
    Precision_label_answer["text"]=pre1
    Recall_label_answer["text"]=rec1
    F1_score_label_answer["text"]=f1_s1
    kq_label_answer["text"] = int(y_pred)
    
    
Variance = tk.Label(window, text = "Variance").place(x = 30, y = 80)

a = tk.Entry(window,textvariable = Variance_entry).place(x = 140, y = 80)

Skewness = tk.Label(window, text = "Skewness").place(x = 30, y = 120)
b = tk.Entry(window, textvariable= Skewness_entry).place(x = 140, y = 120)

Curtosis = tk.Label(window, text = "Curtosis").place(x = 30, y = 160)
c = tk.Entry(window, textvariable= Curtosis_entry).place(x = 140, y = 160)

Entropy = tk.Label(window, text = "Entropy").place(x = 30, y = 200)
d = tk.Entry(window, textvariable=Entropy_entry).place(x = 140, y = 200)



Accuracy_label = tk.Label(window, text = "Accuracy: ").place(x = 30, y = 250)
Accuracy_label_answer = tk.Label(window, text = 'None')

Precision_label = tk.Label(window, text = "Precision: ").place(x = 30, y = 280)
Precision_label_answer = tk.Label(window, text = 'None')

Recall_label = tk.Label(window, text = "Recall").place(x = 30, y = 310)
Recall_label_answer = tk.Label(window, text = 'None')

F1_score_label = tk.Label(window, text = "F1-score:").place(x = 30, y = 340)
F1_score_label_answer = tk.Label(window, text = 'None')

kq_label = tk.Label(window, text = "Dự đoán (Class): ").place(x = 380, y = 160)
kq_label_answer = tk.Label(window, text='None')

submit_button = tk.Button(window,command = submit ,text = 'Submit',width=20, bg="yellow").place(x = 280,  y = 540)
Accuracy_label_answer.place(x = 280, y = 250)

Precision_label_answer.place(x = 280, y = 280)

Recall_label_answer.place(x = 280, y = 310)

F1_score_label_answer.place( x = 280, y = 340)

kq_label_answer.place( x = 480, y = 160)

window.mainloop()