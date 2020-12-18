#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
import pandas as pd
import pyarrow.parquet as pq


# In[2]:


import os, sys
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


# In[3]:


# 구글 드라이브 사용 시 주석 해제
'''
from google.colab import drive
drive.mount('/content/mnt',force_remount=True)

# Path에, 구글드라이브 내의 데이터 경로 설정
path = '/content/mnt/MyDrive/colab_data/big_data/user_vector.parquet'

'''


# In[4]:



# GPU 할당 설정
GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

# Additional Infos
if device.type == 'cuda':
    print(torch.cuda.get_device_name(GPU_NUM))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')


# In[5]:


# Read user_vector.parquet
path = "./data/norm_vector.parquet"
df = pd.read_parquet(path) 


# In[6]:


#device = torch.device('cpu')
# Convert to torch 데이터를 텐서로 불러옴. cpu에 올리는 이유는 램 부족 때문
data = torch.DoubleTensor(df.values).to('cpu')
print("전체 데이터 : ", data.shape)


# In[7]:


def print_news_list_of_user(user_index) :
    base_address = "https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=102&oid={}&aid={}"
    # 참고. 유저가 본 뉴스 목록들
    # user_index에 유저 인덱스 값을 넣으면 된다
    test_user_vector = torch.DoubleTensor(data[user_index])
    print("#"*20)
    print("Commented News")
    for i in range(0, test_user_vector.shape[0]) :
        if test_user_vector[i] > 0 :
            ids = df.columns[i]
            oid = ids[4:7]
            aid = ids[-10:]
            #print(oid,aid)
            print (base_address.format(oid,aid))
    print("#"*20)
    print("Recommended News")
    #pred = model(test_user_vector.float().to(device))
    #pred_val, pred_index = torch.topk(pred,10)
    pred_val, pred_index = get_recommend_news(user_index,500,10)
    for i in pred_index :
        ids = df.columns[i]
        oid = ids[4:7]
        aid = ids[-10:]
        #print(oid,aid)
        print (base_address.format(oid,aid))
    print("#"*20)


# In[8]:


def get_similar_users_vector(main_user_index,num_k) :
    main_user = data[main_user_index].to(device)
    
    # Cosine Similarity 
    cos = nn.CosineSimilarity(dim= -1, eps=1e-6)

    # 높은 Cos값, 인덱스 저장할 리스트
    stored_index = []
    stored_values = []

    # VRAM 부족으로, 10만개씩 끊어서 계산
    # 10만개 당 cos_similarity 높은 num_k개씩 뽑아서 저장
    for i in range(0, int(data.shape[0]/100000)+1) : 
        torch.cuda.empty_cache()
        val = cos(main_user,data[100000 * i : 100000 *i + 100000].to(device))
        values, index = torch.topk(val,num_k)
        index = index + 100000*i
        stored_index.append(index)
        stored_values.append(values)
        torch.cuda.empty_cache()

    ### 최종적으로 num_k개 유저 추리기
    #
    # 텐서로 변환
    stored_index = torch.cat(stored_index,0)
    stored_values = torch.cat(stored_values,0)
    print(stored_index,stored_values)

    # 상위 5개 추리기
    similar_values, index_of_stored_index = torch.topk(stored_values,num_k)

    # 인덱스 변환
    similar_index = []
    for i in index_of_stored_index :
        print(i)
        similar_index.append(stored_index[i])
    ### 최종적으로 num_k개 유저 추리기 끝
    # num_k개 추린 유저들의 벡터를 similar_users에 저장
    similar_users = []
    for i in range(0,num_k) :
      user_id = df.index[similar_index[i]]
      similar_value = similar_values[i]
      print("Index : {:10d}, ID : {:10s}, Similar Value : {}".format(similar_index[i],user_id,similar_value))
      similar_users.append(df.loc[user_id])

    similar_users = torch.DoubleTensor(similar_users)
    return simiilar_users


# In[9]:


def get_recommend_news(main_user_index, num_k, num_news) :
    main_user = data[main_user_index].to(device)
    
    # Cosine Similarity 
    cos = nn.CosineSimilarity(dim= -1, eps=1e-6)

    # 높은 Cos값, 인덱스 저장할 리스트
    stored_index = []
    stored_values = []

    # VRAM 부족으로, 10만개씩 끊어서 계산
    # 10만개 당 cos_similarity 높은 num_k개씩 뽑아서 저장
    for i in range(0, int(data.shape[0]/100000)+1) : 
        torch.cuda.empty_cache()
        val = cos(main_user,data[100000 * i : 100000 *i + 100000].to(device))
        values, index = torch.topk(val,num_k)
        index = index + 100000*i
        stored_index.append(index)
        stored_values.append(values)
        torch.cuda.empty_cache()

    ### 최종적으로 num_k개 유저 추리기
    #
    # 텐서로 변환
    stored_index = torch.cat(stored_index,0)
    stored_values = torch.cat(stored_values,0)

    # 상위 5개 추리기
    similar_values, index_of_stored_index = torch.topk(stored_values,num_k)

    # 인덱스 변환
    similar_index = []
    for i in index_of_stored_index :

        similar_index.append(stored_index[i])
    ### 최종적으로 num_k개 유저 추리기 끝
    # num_k개 추린 유저들의 벡터를 similar_users에 저장
    similar_users = []
    for i in range(0,num_k) :
      user_id = df.index[similar_index[i]]
      similar_value = similar_values[i]
      #print("Index : {:10d}, ID : {:10s}, Similar Value : {}".format(similar_index[i],user_id,similar_value))
      similar_users.append(df.loc[user_id])

    similar_users = torch.DoubleTensor(similar_users)
        # num_k명의 벡터 평균치 계산
    similar_users_mean = torch.mean(similar_users,dim=0)

    # 하나의 평균 벡터가 나옴
    similar_users_mean.shape

    # num_k명의 평균 벡터와, 주인공 유저와의 벡터 차잇값을 계산
    difference_val = similar_users_mean - main_user.cpu()
    
    # 차이가 가장 많이 나는 값(뉴스) 추출. 차잇값, 인덱스 반환
    num_recommend_news = num_news  # 추천할 뉴스의 갯수
    recommend_values, recommend_news_index = torch.topk(difference_val,num_recommend_news)
    return recommend_values, recommend_news_index


# In[10]:


# 위의 과정들을 통해, 뉴럴넷에서 학습시킬 Y값 텐서를 생성
def get_ys(user_index_start,user_index_end) :
    ret = []
    user_num = user_index_end - user_index_start
    for i in range(user_index_start,user_index_end) :
        
        if i%10 == 0 :
            print("get_ys : user index :{} / {}".format(i,user_num))
        val, index = get_recommend_news(i,500,10)
        y_vec = torch.zeros((data.shape[1]))
        for enum,idx in enumerate(index) :
            y_vec[idx] = val[enum]
        ret.append(y_vec)
    ret = torch.reshape(torch.cat(ret),(user_num,1422))
    return ret


# In[11]:


def get_y(user_idx) :
    val, index = get_recommend_news(user_idx,500,10)
    y_vec = torch.zeros((data.shape[1]))
    for enum,idx in enumerate(index) :
        y_vec[idx] = val[enum]
    return y_vec


# In[12]:


##########
# 아래는 위의 과정들을, 뉴럴넷으로 학습시키는거
# 1422 차원의 유저 벡터 들어오면,
# 1422 차원의 뉴스 벡터 나오는 뉴럴넷



class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        생성자에서 2개의 nn.Linear 모듈을 생성(Instantiate)하고, 멤버 변수로
        지정합니다.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        순전파 함수에서는 입력 데이터의 Variable을 받아서 출력 데이터의 Variable을
        반환해야 합니다. Variable 상의 임의의 연산자뿐만 아니라 생성자에서 정의한
        모듈을 사용할 수 있습니다.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

    

class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        """
        생성자에서 2개의 nn.Linear 모듈을 생성(Instantiate)하고, 멤버 변수로
        지정합니다.
        """
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        """
        순전파 함수에서는 입력 데이터의 Variable을 받아서 출력 데이터의 Variable을
        반환해야 합니다. Variable 상의 임의의 연산자뿐만 아니라 생성자에서 정의한
        모듈을 사용할 수 있습니다.
        """
        h1_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h1_relu).clamp(min=0)
        y_pred = self.linear3(h2_relu)
        return y_pred

    
    
    


# In[13]:


'''
linear1 = torch.nn.Linear(1422, 2844, bias=True)
linear2 = torch.nn.Linear(2844, 1800, bias=True)
linear3 = torch.nn.Linear(1800, 1422, bias=True)
relu = torch.nn.ReLU()

# Initialization
torch.nn.init.normal_(linear1.weight)
torch.nn.init.normal_(linear2.weight)
torch.nn.init.normal_(linear3.weight)

# model
model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3).to(device)
# define cost/loss & optimizer
#criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
criterion = torch.nn.MSELoss(size_average=False).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

#optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
'''


# In[14]:


# 데이터 크롭. 전체 유저 학습하기엔 시간부족..
# 0부터 1000까지의 유저에 대해서만 학습

# 데이터를 로드할건지, 새로 만들건지
# 학습시킬 유저의 인덱스 값
USER_NUM = 10000
try :
    y = torch.load("./data_y/norm_y_to_{}.pth".format(USER_NUM)).to(device)
    print("Load Y Values to {} Completed".format(USER_NUM))
except :
    print("Create Y Values to {}".format(USER_NUM))
    y = get_ys(0,USER_NUM)
    torch.save(y,"./data_y/norm_y_to_{}.pth".format(USER_NUM))
    
    
x = data[0:USER_NUM].float().to(device)
y = y.to(device)


# In[15]:


# N은 배치 크기이며, D_in은 입력의 차원입니다;
# H는 은닉 계층의 차원이며, D_out은 출력 차원입니다:
N, D_in, H, D_out = 1, data.shape[1], 2000, data.shape[1]
H1, H2 = 3000,2000
# 입력과 출력을 저장하기 위해 무작위 값을 갖는 Tensor를 생성하고, Variable로
# 감쌉니다.
# 앞에서 정의한 클래스를 생성(Instantiating)해서 모델을 구성합니다.
#model = TwoLayerNet(D_in, H, D_out)

# 손실함수와 Optimizer를 만듭니다. SGD 생성자에서 model.parameters()를 호출하면
# 모델의 멤버인 2개의 nnLinear 모듈의 학습 가능한 매개변수들이 포함됩니다.

# 모델 넘버 지정. 데이터에 있으면 로드, 없으면 새로 만듬

MODEL_NUM = 11
model = None
try :
    model = torch.load("./model/model_{}.pth".format(MODEL_NUM))
    model.eval()
    print("Load Model {} Completed".format(MODEL_NUM))
except :
    if MODEL_NUM == 6 :
        model = ThreeLayerNet(D_in, H1, H2, D_out)
    
    else :
        model = TwoLayerNet(D_in, H, D_out)
        
    print("New Model {} Created".format(MODEL_NUM))

criterion = torch.nn.MSELoss(size_average=False).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
model.to(device)


# In[16]:


x.shape


# In[17]:


y.shape


# In[18]:


def test(start,end) :
    results = []
    test_len = end - start
    for i in range(start,end) :
        if (i-start) % (test_len / 100) == 0 :
            print("testing... ",i)
        test1_val, test1_index = get_recommend_news(i,500,10)
        test2_val, test2_index = torch.topk(model(data[i].cuda().float()),10)
        correct = 0
        for j in test1_index.to('cpu') :
            if j in test2_index.to('cpu') :
                correct +=1
        results.append(correct/10)
    results = torch.tensor(results)
    print(results.mean())
    return results


# In[ ]:





# In[ ]:


iteration = 0
#print("DATA TO : {}, MODEL NUM : {}".format(USER_NUM,MODEL_NUM))
#y = torch.where(x==0, y , torch.ones(y.shape).to(device) )

while (True) :
    # 순전파 단계: 모델에 x를 전달하여 예상하는 y 값을 계산합니다.
    #print(x.dtype)
    iteration += 1
    y_pred = model(x)

    # 손실을 계산하고 출력합니다.
    #print(y_pred.dtype,y.dtype)
    loss = criterion(y_pred, y)

    # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신합니다.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if iteration % 100 == 0:
        print("{:5d}, loss : {:.7f}".format(iteration, loss.item()))
    
    if iteration % 1000 == 0 :
        test(19000,19010)
        print("test")
    if iteration % 10000 == 0 :
        torch.save(model,"./model/model_{}.pth".format(MODEL_NUM))
        
    if loss < 0.1 : 
        print("Loss : {}, Learning exit".format(loss.item()))
        break


# In[ ]:


# 모델 저장
torch.save(model,"./model/model_{}.pth".format(MODEL_NUM))


# In[ ]:


test1_val, test1_index = get_recommend_news(20002,500,10)
test2_val, test2_index = torch.topk(model(data[20002].cuda().float()),10)
print("기존 알고리즘이 추천하는 뉴스 인덱스 : \n{}, \n 추천값:{}".format(test1_index, test1_val))
print("뉴럴넷이 추천하는 뉴스 인덱스 : \n{},\n 추천값:{}".format(test2_index, test2_val))


# In[ ]:


print_news_list_of_user(25016)


# In[ ]:


def test(start,end) :
    results = []
    for i in range(start,end) :
        test1_val, test1_index = get_recommend_news(i,500,10)
        test2_val, test2_index = torch.topk(model(data[i].cuda().float()),10)
        correct = 0
        for j in test1_index.to('cpu') :
            if j in test2_index.to('cpu') :
                correct +=1
        results.append(correct/10)
    results = torch.tensor(results)
    print(results.mean())
    return results


# In[ ]:


test(29000,29050)


# In[ ]:


results = torch.tensor(results)


# In[ ]:


print(results.mean())

