import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics #importing modul metrik

# Memuat dataset
data = pd.read_csv("E:\Semester 4\data mining/bc.csv",encoding = "ISO-8859-1")
print(data.head(5))

# ringkasan dataset
data.info()

#menghapus kolom yang tidak berguna
#menghapus kolom "id"
data.drop("id",axis=1,inplace=True)

#menghapus the "Unnamed: 32" column
data.drop("Unnamed: 32",axis=1,inplace=True)

#hasil
data.info()

# 5 baris pertama
print(data.head(5))

#diagnosis adalah variabel yang bertanggung jawab untuk klasifikasi
#mengganti M dan B masing-masing dengan 1 dan 0
data.diagnosis=data.diagnosis.map({'M':1,'B':0})

#menghitung variabel diagnosis
data.diagnosis.value_counts()

# preprocessing datasetselesai
#splitting dataset ke training dan testing
train, test = train_test_split(data, test_size = 0.3,random_state=1234)

#mencari hasil
print(train.shape)
print(test.shape)

#membuat variabel independen untuk training
train_X = train.iloc[:, 1:31]
#membuat variabel responsible untuk training
train_y=train.diagnosis
#membuat variabel independen untuk testing
test_X= test.iloc[:, 1:31]
#membuat variabel responsible untuk ttesting
test_y =test.diagnosis

#mencari hasil
print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)
neighbors=np.arange(1,31)
accuracy_train=[]
accuracy_test=[]
for i,k in enumerate(neighbors):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_X,train_y)
    accuracy_train.append(knn.score(train_X,train_y))
    accuracy_test.append(knn.score(test_X,test_y))

plt.figure(figsize=(13,8))
plt.plot(neighbors,accuracy_train,label="Akurasi training")
plt.plot(neighbors,accuracy_test,label="Akurasi testing")
acideal=np.max(accuracy_test)
kideal=1+accuracy_test.index(np.max(accuracy_test))
plt.title('Tingkat Akurasi')
plt.xlabel("K yang dipilih")
plt.ylabel("Tingkat Akurasi")
plt.xticks(neighbors)
plt.legend()
plt.show()
print("Akurasi ideal:",acideal)
print("K ideal:",kideal)

#membuat instance
model = KNeighborsClassifier(n_neighbors=int(input("Masukan jumlah k:")))
#learning
model.fit(train_X,train_y)
#Prediksi
prediction=model.predict(test_X)

#evaluation (Akurasi)
print("Akurasi:",metrics.accuracy_score(prediction,test_y))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))

datatest=pd.DataFrame(test_X)
datatest['diagnosis']=test_y
datatest['prediksi']=prediction
print(datatest)