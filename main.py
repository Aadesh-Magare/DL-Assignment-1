
"""

  Deep Learning E0250: Assignment 1
  Author: Magare Aadesh Gajanan
  Packages: Python 3.7, pytorch 1.4, sklearn.

"""
#%% 
import sys
import torch
# print(torch.__version__)
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
torch.manual_seed(111)
np.random.seed(111)

#%% Encode correct labels
def fizzbuzz(n):
  if not (n % 15):
    return 0 #'fizzbuzz'
  elif not (n % 3):
    return 1 #'fizz'
  elif not (n % 5):
    return 2 #'buzz'
  else:
    return 3 # original number

#%% Encode numbers in binary of size 10 bits.
def binary_encode(x):
  return list(map(lambda n: list(map(int, format(n, '010b'))), x))

#%% Define model architecture, train and save model
def train_model(filepath):
  x_train = list(np.arange(101, 1001))
  y_train = Variable(torch.tensor(list(map(lambda x: fizzbuzz(x), x_train))))
  x_train = Variable(torch.tensor(binary_encode(x_train), dtype=torch.float32))

  xt =  list(np.arange(1, 101))
  y_test = Variable(torch.tensor(list(map(lambda x: fizzbuzz(x), xt))))
  x_test = Variable(torch.tensor(binary_encode(xt), dtype=torch.float32))

  # print(y_train.shape)
  # print(x_train.shape)
  # print(x_train[:5])
  # print(y_train[:5])
  # print(x_test[:5])

  model = torch.nn.Sequential(
      torch.nn.Linear(10, 10),
      torch.nn.Tanh(),
      torch.nn.Linear(10, 8),
      torch.nn.Tanh(),
      # torch.nn.Linear(8, 8),
      # torch.nn.Tanh(),
      torch.nn.Linear(8, 4)
      # ,torch.nn.Softmax()
  )
  loss_function = torch.nn.CrossEntropyLoss(reduction='none')
  learning_rate = 1e-3
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  epocs = 10000


  for i in range(epocs):
    preds = model(x_train)
    # print(preds)
    loss = loss_function(preds, y_train).mean()
    if not(i % 500):
      print('Loss: ', loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  preds = model(x_test)
  preds = torch.argmax(preds, axis=1)
  print ([output_labels(i, x) for i, x in zip(xt, preds)])
  print('Accuracy:' , accuracy_score(preds, y_test))

  torch.save(model, 'model/fizzbuzz_trained')
  print('Saved trained model to: model/fizzbuzz_trained')
#%% Load saved model for inference.
def test_model(filepath):
  model = torch.load('model/fizzbuzz_trained')
  x = list(map(int, open(filepath).read().split()))
  y_test = list(map(lambda xx: fizzbuzz(xx), x))
  y_test = [output_labels(i, y) for i, y in zip(x, y_test)]

  x_test = torch.tensor(binary_encode(x), dtype=torch.float32)
  preds = model(x_test)
  preds = torch.argmax(preds, axis=1)
  outputs = [output_labels(i, y) for i, y in zip(x, preds)]

  with open('Software1.txt', 'w') as f:
    # f.write('\n'.join(map(str, y_test)))
    for y in y_test:
      f.write(f'{str(y)}\n')

  with open('Software2.txt', 'w') as f:
    # f.write('\n'.join(map(str, outputs)))
    for o in outputs:
      f.write(f'{str(o)}\n')

  print('Respones saved to files Software1.txt and Software2.txt')

#%% Decode fizzbuzz output
def output_labels(i, x):
  label = ['fizzbuzz', 'fizz', 'buzz', i]
  return label[x]

#%% Parse arguments and run.
if __name__ == '__main__':
  if len(sys.argv) > 1:
    if sys.argv[1] == '--train-data':
      train_model(sys.argv[2])
    elif sys.argv[1] == '--test-data':
      test_model(sys.argv[2])
    else:
      print('''Program Usage:
          Train: python main.py --train-data <train-data-path>
          Test:  python main.py --test-data <test-data-path>
      ''')
  else:
    print('Missing required parameters')

#%%
