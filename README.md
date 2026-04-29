# Developing-a-Neural-Network-Classification-Model

## 🎯 AIM  
To design, implement, and evaluate a neural network classification model for predicting iris flower species based on input features.

---

## 📚 THEORY  
Classification is a supervised learning task where the goal is to assign input data into predefined categories.

The **Iris dataset** is a classic benchmark dataset in machine learning. It contains:
- 150 samples  
- 3 classes: *Iris setosa*, *Iris versicolor*, *Iris virginica*  
- 4 features:
  - Sepal length  
  - Sepal width  
  - Petal length  
  - Petal width  

A neural network model learns patterns in these features to classify flowers into their correct species.

Unlike regression, classification models output probabilities and assign the class with the highest probability.

> 💡 **One-line takeaway:**  
> Classification = predicting categories (labels) instead of numbers.

---



## ⚙️ DESIGN STEPS  

### 🔹 STEP 1: Load Dataset  
Load the Iris dataset using `sklearn.datasets`.

### 🔹 STEP 2: Data Preprocessing  
- Check for missing values  
- Normalize features using `StandardScaler`

### 🔹 STEP 3: Train-Test Split  
Split dataset into training and testing sets (e.g., 80:20 ratio).

### 🔹 STEP 4: Model Training  
- Define neural network using PyTorch  
- Use CrossEntropyLoss for classification  
- Train using an optimizer like SGD or Adam  

### 🔹 STEP 5: Model Evaluation  
Evaluate model performance using test data.

### 🔹 STEP 6: Performance Metrics  
- Accuracy score  
- Confusion matrix  
- Classification report (precision, recall, F1-score)

---

## 💻 PROGRAM  

**Name: Prajin S **  
**Register Number: 212223230151**  

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import TensorDataset,DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target
df=pd.DataFrame(X,columns=iris.feature_names)
df['target']=y
df.tail()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train=torch.tensor(X_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.long)
y_test=torch.tensor(y_test,dtype=torch.long)
train_data=TensorDataset(X_train,y_train)
test_data=TensorDataset(X_test,y_test)
train_loader=DataLoader(train_data,batch_size=16,shuffle=True)
test_loader=DataLoader(test_data,batch_size=16)
class IrisClassifier(nn.Module):
  def __init__(self,input_size):
    super(IrisClassifier,self).__init__()
    self.fc1=nn.Linear(input_size,16)
    self.fc2=nn.Linear(16,8)
    self.fc3=nn.Linear(8,3)

  def forward(self,x):
    x= F.relu(self.fc1(x))
    x= F.relu(self.fc2(x))
    return self.fc3(x)
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()
    if(epoch+1)%10==0:
      print(f'Epoch [{epoch+1}/{epochs}], Loss:{loss.item():.4f}')
model=IrisClassifier(input_size=X_train.shape[1])
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
train_model(model,train_loader,criterion,optimizer,epochs=100)
model.eval()
predictions,actuals=[],[]
with torch.no_grad():
  for X_batch,y_batch in test_loader:
    outputs=model(X_batch)
    _,predicted=torch.max(outputs,1)
    predictions.extend(predicted.numpy())
    actuals.extend(y_batch.numpy())
accuracy=accuracy_score(actuals,predictions)
conf_matrix=confusion_matrix(actuals,predictions)
class_report=classification_report(actuals,predictions,target_names=iris.target_names)
print(f'Test Accuracy: {accuracy:.2f}\n')
print("Classification Report:\n",class_report)
print("Confusion Matrix:\n",conf_matrix)
sns.heatmap(conf_matrix,annot=True,cmap='Blues',xticklabels=iris.target_names,yticklabels=iris.target_names,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
sample_input=X_test[5].unsqueeze(0)
with torch.no_grad():
  output=model(sample_input)
  predicted_class_index=torch.argmax(output[0]).item()
  predicted_class_label=iris.target_names[predicted_class_index]
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {iris.target_names[y_test[5].item()]}')
```

## 📈 OUTPUT
<img width="714" height="231" alt="image" src="https://github.com/user-attachments/assets/25b3b1d7-6559-4f56-b34b-160cfda84724" />

<img width="763" height="242" alt="image" src="https://github.com/user-attachments/assets/cfe459d1-3cc8-44e4-8a9e-2a83967eecf4" />

<img width="323" height="217" alt="image" src="https://github.com/user-attachments/assets/4b0e142f-c5f9-40c2-9d0e-f6f2cedfc6d7" />

<img width="550" height="270" alt="image" src="https://github.com/user-attachments/assets/1e9384d8-351d-4d67-a716-4c28a641d8a7" />

<img width="683" height="662" alt="image" src="https://github.com/user-attachments/assets/b9f294c9-49d8-4831-af13-d5d5acec6945" />

## 🧪 OBSERVATIONS
The model effectively distinguishes between the three iris species
High accuracy is achieved due to well-separated features
Minimal misclassification occurs between similar classes
## ✅ RESULT

A neural network classification model was successfully developed using PyTorch. The model accurately classified iris flower species based on input features and achieved strong performance on test data.

## 🚀 FUTURE ENHANCEMENTS
Use deeper neural networks for complex datasets
Apply dropout to prevent overfitting
Try different optimizers like Adam
Extend to real-world classification problems

