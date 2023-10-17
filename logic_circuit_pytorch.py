import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#データ生成 （本当は[0,0]の時のANDは0だが、学習時の都合で-1としている）
train = np.array([])
label = np.array([])
for i in range(1000):
  num = np.random.randint(0,4)
  if num==0:
    train = np.append(train, [0,0])
    label = np.append(label, [-1])
  if num==1:
    train = np.append(train, [0,1])
    label = np.append(label, [1])
  if num==2:
    train = np.append(train, [1,0])
    label = np.append(label, [1])
  else:
    train = np.append(train, [1,1])
    label = np.append(label, [1])
train = np.reshape(train, (-1, 2))
label = np.reshape(label, (-1, 1))

#ハイパーパラメータなどの設定値
num_epochs = 50         # 学習を繰り返す回数
num_batch = 100         # 一度に処理するデータの数
learning_rate = 0.01        # 学習率

#numpy→torchに変換
train = torch.from_numpy(train.astype(np.float32)).clone()
label = torch.from_numpy(label.astype(np.float32)).clone()
#データセット
train_dataset = TensorDataset(train, label)
train_loader = DataLoader(train_dataset, batch_size=num_batch, shuffle=True)
#データローダー
test_dataset = TensorDataset(train, label)
test_loader = DataLoader(test_dataset, batch_size=num_batch)

#モデルの定義
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.fc = nn.Linear(2,1)
    
  def forward(self, input):
    return self.fc(input)
  
  model = Model() #モデルの呼び出し
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #最適化関数
loss_function = nn.MSELoss() #損失関数

print("EPOCH.     LOSS.  ACCURACY")
#学習開始
for epoch in range(1, num_epochs+1, 1):
  #学習
  train_losses = np.array([])
  model.train() 
  for batch in train_loader:
    x, t = batch               #訓練データをxに、正解データをtに取り出す
    optimizer.zero_grad()      #optimizerを初期化
    y = model(x)              #ニューラルネットワークの処理
    loss = loss_function(y, t)  #損失関数の計算
    loss.backward()            #勾配の計算
    optimizer.step()            #重みの更新
    loss = loss.detach().numpy().copy()
    train_losses = np.append(train_losses, loss)

  #評価
  model.eval() 
  valid_losses = np.array([])
  for tbatch in test_loader:
    xval, tval = tbatch                   #評価用データをxvalに、正解データをtvalに取り出す
    with torch.set_grad_enabled(False):  #重みが更新されないように
      yval = model(xval)
    y_label = torch.where(yval>0, 1, -1) #yvalが0以上なら1、0以下なら-1に変換
    accuracy = (y_label == tval).sum() / len(tval) #正答率の計算

  print('{:>4}  　　　{:^10.4f}  {:^10.4f}'.format(epoch, np.mean(train_losses), accuracy))
