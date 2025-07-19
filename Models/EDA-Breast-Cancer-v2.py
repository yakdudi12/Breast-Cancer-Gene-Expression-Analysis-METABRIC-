import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

df = pd.read_csv(r'...', encoding='utf-8',low_memory=False)
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999
df_filtrado = df.iloc[:, np.r_[1,4,5,7,20,23,24,29, 31:693]]

#print(df_filtrado.dtypes)
#Cleaning the dataset
df_filtrado = df_filtrado.drop_duplicates()
por_na = (df_filtrado.isna().sum() / len(df_filtrado)) * 100
print(por_na.sort_values(ascending=False).head())

df_filtrado = df_filtrado.dropna()
df_filtrado = df_filtrado.reset_index(drop=True)

#One-hot encoding; FOR DISPLAY ONLY, not for use in deep learning
df_filtrado_onehot = pd.get_dummies(df_filtrado,columns=['cancer_type_detailed','cellularity','pam50_+_claudin-low_subtype'])

#Matrix correlation:
correlation_numeric = df_filtrado_onehot.corr(numeric_only=True)

correlation_numeric_pairs = correlation_numeric.unstack()

# We eliminate autocorrelations (correlation of a variable with itself = 1)
sorted_pairs = correlation_numeric_pairs.sort_values(kind="quicksort",ascending=False)

# Optional: remove duplicates (because corr(A,B) = corr(B,A))
sorted_pairs = sorted_pairs[sorted_pairs.index.get_level_values(0) < sorted_pairs.index.get_level_values(1)]

#Top 10
print(sorted_pairs.head(15))

df_mutation_predictor = df_filtrado.iloc[:,8:]

#Transformation to binary encoding columns of "0" or "1" mutations
mut_columns2 = [c for c in df_mutation_predictor.columns if c.endswith('_mut')]

for col in mut_columns2:
    df_mutation_predictor[col] = df_mutation_predictor[col].astype(str).str.strip()
    df_mutation_predictor[col] = (df_mutation_predictor[col] != '0').astype(int)

#print(df_mutation_predictor.sample(1))

#We started developing the model:
import sklearn
import sklearn.model_selection
df_train,df_test = sklearn.model_selection.train_test_split(df_mutation_predictor,train_size=0.8)
train_stats = df_train.describe().transpose()
#print(train_stats)
needtostandar_columns = df_mutation_predictor.select_dtypes(include=np.float64).columns
#print(needtostandar_columns)
df_train_norm,df_test_norm = df_train.copy(),df_test.copy()

for i in needtostandar_columns:
    '''Making sure it's 100% float'''
    df_test_norm[i] = df_test_norm[i].astype(float)
    df_train_norm[i] = df_train_norm[i].astype(float)

    mean = train_stats.loc[i, 'mean']
    std = train_stats.loc[i, 'std']
    '''Standardization of both'''
    df_train_norm.loc[:, i] = (df_train_norm.loc[:,i] - mean)/std
    df_test_norm.loc[:, i] = (df_test_norm.loc[:, i] - mean)/std

#Formatting in Pytorch Tensors:
#print(df_train_norm.dtypes)

x_train = torch.tensor(df_train_norm.select_dtypes(include=np.float64).values).float()
x_test = torch.tensor(df_test_norm.select_dtypes(include=np.float64).values).float()

y_train = torch.tensor(df_train_norm.select_dtypes(include=np.int64).values).float()
y_test = torch.tensor(df_test_norm.select_dtypes(include=np.int64).values).float()

###Keep the 10 best represented classes###
topk = 20
top_idx = torch.topk(y_train.sum(0), k=topk).indices
y_train = y_train[:, top_idx]
y_test = y_test[:, top_idx]

###Filters to combat imbalance###
y_sums = y_train.sum(axis=0)
mask = y_sums >= 50
y_train_filtered = y_train[:, mask]
y_test_filtered = y_test[:, mask]

y_train = y_train_filtered
y_test = y_test_filtered

#Create the DataLoader:
from torch.utils.data import DataLoader, TensorDataset
traind_ds = TensorDataset(x_train,y_train)
batch_size = 100
traind_dl = DataLoader(traind_ds,batch_size=batch_size,shuffle=True)

#Building the architecture of the Model:
'''First test architecture: Sequential 4 layers'''
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

#Instantiate the model and select loss and optimizer:
mut_cols_filtradas = df_train_norm.select_dtypes(include=np.int64).columns.tolist()

from sklearn.metrics import classification_report
#Accumulators
recalls_por_clase = {col: [] for col in mut_cols_filtradas}
f1_por_clase = {col: [] for col in mut_cols_filtradas}

num_runs = 100 #Number of runs
for run in range(num_runs):
    print(f"\n=== Run {run+1}/{num_runs} ===")

    modelA = MultilabelMLP(input_dim=x_train.shape[1], output_dim=y_train.shape[1])
    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(modelA.parameters(), lr=1e-3, weight_decay=1e-5)

    num_epoch = 300
    train_losses = []
    train_accuracies = []


    for epoch in range(num_epoch):
        modelA.train()
        optimizer.zero_grad()
        logits = modelA(x_train)
        loss_value = loss(logits, y_train)
        loss_value.backward()
        optimizer.step()

    modelA.eval()
    with torch.no_grad():
        logits_test = modelA(x_test)
        probs = torch.sigmoid(logits_test)
        preds_test = (probs > 0.5).int()

    # classification_report in diccionary:
    report = classification_report(y_test.cpu().numpy(), preds_test.cpu().numpy(), zero_division=0, output_dict=True)

    #Saving:
    for i, col in enumerate(mut_cols_filtradas):
        if str(i) in report:
            recalls_por_clase[col].append(report[str(i)]['recall'])
            f1_por_clase[col].append(report[str(i)]['f1-score'])

print("\n=== Avg per class ===")
recallss = []
f1_scoress = []
columnass = []
for col in mut_cols_filtradas:
    if len(recalls_por_clase[col]) > 0:
        mean_recall = np.mean(recalls_por_clase[col])
        mean_f1 = np.mean(f1_por_clase[col])
        recallss.append(np.mean(recalls_por_clase[col]))
        f1_scoress.append(np.mean(f1_por_clase[col]))
        columnass.append(col)
        print(f'{col}: recall={mean_recall:.3f}, f1={mean_f1:.3f}')

#To dataframe:
df_modelA_metrics = pd.DataFrame({'Mutacion':columnass,'Recall':recallss,
                                  'F1-score':f1_scoress})
df_modelA_metrics = df_modelA_metrics.sort_values('F1-score',ascending=False).head(12)
# Plot:
plt.figure(figsize=(12, 6))
sns.barplot(data=df_modelA_metrics,x='Mutacion',y='F1-score',hue='Recall')
plt.xticks(rotation=45)
plt.show()





