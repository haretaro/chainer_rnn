
# coding: utf-8

# In[414]:

get_ipython().magic('matplotlib inline')
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.cuda
from chainer import optimizers
from chainer import serializers
from chainer import cuda
import numpy as np
import matplotlib.pyplot as plt


# In[415]:

n_epoch = 200
batchsize = 100
bprop_len = 10
n_units = 30
grad_clip = 5
use_gpu = False


# In[416]:

xp = cuda.cupy if use_gpu is True else np


# In[417]:

t = np.arange(0,1000,0.1, dtype=np.float32)

train_data = np.sin(t) * np.sin(5 * t)
#train_data = np.asarray([1 if (i/5)%2 == 0 else -1 for i in range(10000)], dtype=np.float32)
#train_data = np.asarray([1 if i%10 < 5 else -1 for i in range(10000)], dtype=np.float32) #10ユニットもあれば十分
#train_data = np.asarray([1 if x > 1 else x for x in 1.5 * np.sin(t)], dtype=np.float32)
#train_data = t % 3
train_data = np.asarray([x.round(0) for x in 2 * np.sin(t)], dtype=np.float32)
plt.plot(train_data[:100])


# In[418]:

class RNN(chainer.Chain):
    def __init__(self, n_units):
        super(RNN, self).__init__(
            l1 = L.Linear(1, n_units),
            l2 = L.LSTM(n_units, n_units),
            l3 = L.Linear(n_units, 1)
        )
    
    def __call__(self,x,t):
        return F.mean_squared_error(self.predict(x),t)
        
    def reset_state(self):
        self.l2.reset_state()
    
    def predict(self,x):
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        y = self.l3(h2)
        return y


# In[419]:

#ネットワークを試す関数
def evaluate(model,num,origin=0):
    t = chainer.Variable(xp.array([[origin]],dtype=np.float32))
    output = []
    evaluator = model.copy()
    evaluator.reset_state()
    for i in range(num):
        t = evaluator.predict(t)
        output.append(t.data[0])
    return output


# In[420]:

model = RNN(n_units)
if use_gpu is True:
    model.to_gpu()
#optimizer = optimizers.SGD(lr=1.)
optimizer = optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))
loss = []
length = train_data.shape[0]
jump = length // batchsize
batch_idxs = list(range(batchsize))
accum_loss = 0
epoch = 0
loss_data = 0
for i in range(jump * n_epoch):
    x = chainer.Variable(xp.asarray([[train_data[(jump * j + i) % length]] for j in batch_idxs]))
    t = chainer.Variable(xp.asarray([[train_data[(jump * j + i + 1) % length]] for j in batch_idxs]))
    loss_i = model(x,t)
    accum_loss += loss_i
    loss_data += accum_loss.data
        
    if (i+1) % jump == 0:
        epoch += 1
        #if epoch > 5:
        #    optimizer.lr /= 1.3
        print('epoch {}, error {}'.format(epoch, loss_data / length))
        loss.append(loss_data / length)
        loss_data = 0
        
    
    if (i+1) % bprop_len == 0:
        model.zerograds()
        accum_loss.backward()
        accum_loss.unchain_backward()
        accum_loss = 0
        optimizer.update()


# In[421]:

plt.plot(loss)
plt.title('mean square error')


# In[426]:

output = evaluate(model,400,origin=0)
plt.plot(output[:100])


# In[427]:

plt.plot(output[300:400])


# In[ ]:




# In[ ]:




# In[ ]:



