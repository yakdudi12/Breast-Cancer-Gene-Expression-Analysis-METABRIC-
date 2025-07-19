'''Predictions RNA-seq per Cancer Type'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

df = pd.read_csv(r'...', encoding='utf-8',low_memory=False)
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999
#print(df.dtypes)
df_filtrado = df.iloc[:, np.r_[4, 31:520]]
#print(df_filtrado.dtypes)
#Check the data:
plt.figure(figsize=(12,6))
sns.barplot(data=df_filtrado['cancer_type_detailed'],palette='hls')
plt.show()

df_filtrado = df_filtrado.drop_duplicates()
por_na = (df_filtrado.isna().sum() / len(df_filtrado)) * 100
print(por_na.sort_values(ascending=False).head())
df_filtrado = df_filtrado.dropna()
df_filtrado = df_filtrado.reset_index(drop=True)
print(df_filtrado.sample(1))

#Determine the objective:
target_colum = 'cancer_type_detailed'

### Correction of underrepresented classes###
class_counts = df_filtrado[target_colum].value_counts()
valid_classes = class_counts[class_counts >= 50].index
df_filtrado = df_filtrado[df_filtrado[target_colum].isin(valid_classes)].reset_index(drop=True)

y = df_filtrado[target_colum]
x = df_filtrado.drop(columns=[target_colum])

y_encoded, cancer_type_categories = pd.factorize(y) #codificar la columna
print("Categories Cancer Type:", list(cancer_type_categories))

print(df_filtrado[target_colum].value_counts())


#We started developing the model:
import sklearn
import sklearn.model_selection
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y_encoded, train_size=0.8, stratify=y_encoded
)

train_stats = X_train.describe().transpose()
for i in X_train.columns:
    mean = train_stats.loc[i, 'mean']
    std = train_stats.loc[i, 'std']
    '''Standardization'''
    X_train.loc[:, i] = (X_train.loc[:,i] - mean)/std
    X_test.loc[:, i] = (X_test.loc[:, i] - mean)/std

#convert to tensors:
x_train = torch.tensor(X_train.values, dtype=torch.float32)
x_test = torch.tensor(X_test.values, dtype=torch.float32)

#One-hot encoding:
num_classes = len(cancer_type_categories)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print("Final dimensions")
print("x_train:", x_train.shape)
print("y_train_tensor:", y_train_tensor.shape)

#Creating the dataloader:
from torch.utils.data import DataLoader, TensorDataset
traind_ds = TensorDataset(x_train,y_train_tensor)
batch_size = 100
traind_dl = DataLoader(traind_ds,batch_size=batch_size,shuffle=True)

#Building the model:
class MultilabelMLP(nn.Module):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0,3),

            nn.Linear(256,output_dim)
        )

    def forward(self,x):
        return self.net(x)

#Instance the model and select loss and optimizer:
modelA = MultilabelMLP(input_dim=x_train.shape[1], output_dim=num_classes)
optimizer = torch.optim.Adam(modelA.parameters(), lr=1e-3, weight_decay=1e-5)
loss = nn.CrossEntropyLoss()
num_epoch = 300
train_losses = []
train_accuracies = []

#Training Loop:
for epoch in range(num_epoch):
    modelA.train()
    optimizer.zero_grad()
    logits = modelA(x_train)
    loss_value = loss(logits, y_train_tensor)
    loss_value.backward()
    optimizer.step()

    #Model evaluation:
    with torch.no_grad():
        logits = modelA(x_test)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_test_tensor).float().mean().item()
    train_losses.append(loss_value.item())
    train_accuracies.append(acc)
    print(f"Epoch [{epoch + 1}/{num_epoch}] Loss: {loss_value.item():.4f} Acc: {acc:.4f}")

modelA.eval()
with torch.no_grad():
    logits_test = modelA(x_test)
    preds_test = torch.argmax(logits_test, dim=1)

fig = plt.figure(figsize=(12, 4))
# Loss
ax = fig.add_subplot(1, 2, 1)
plt.plot(train_losses, lw=2, label='Train loss',color='red')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.title('Training Loss', fontsize=14)
# Accuracy
ax = fig.add_subplot(1, 2, 2)
plt.plot(train_accuracies, lw=2, label='Train Accuracy',color='red')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.title('Training Accuracy', fontsize=14)
plt.tight_layout()
plt.show()

#Another metric
from sklearn.metrics import classification_report
report = classification_report(y_test_tensor.cpu().numpy(), preds_test.cpu().numpy(), zero_division=0,output_dict=True)
print(classification_report(y_test_tensor.cpu().numpy(), preds_test.cpu().numpy(), zero_division=0))

#Plot
recalls_por_clase = []
f1_por_clase = []
cols = []
# Storing metrics:
for i, col in enumerate(cancer_type_categories):
    if str(i) in report:
        recalls_por_clase.append(report[str(i)]['recall'])
        f1_por_clase.append(report[str(i)]['f1-score'])
        cols.append(col)

df_model_metrics = pd.DataFrame({'Cancer Type':cols,'Recall':recalls_por_clase,
                                 'F1-score':f1_por_clase})

df_long = pd.melt(df_model_metrics, id_vars='Cancer Type', value_vars=['Recall', 'F1-score'],
                  var_name='Metric', value_name='Score')
plt.figure(figsize=(12,6))
sns.barplot(data=df_long, x='Cancer Type', y='Score', hue='Metric',palette='flare')
plt.title('Recall y F1-score per class Cancer Type')
plt.tight_layout()
plt.grid(True)
plt.show()