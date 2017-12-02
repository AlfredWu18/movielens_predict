import gzip
from collections import defaultdict
import numpy as np
import random

def train_para(data, lam):
    alpha = 0.0
    beta_u = np.zeros(len(users))
    beta_i = np.zeros(len(businesses))
    gamma_u = np.ones(len(users))*1
    gamma_i = np.ones(len(businesses))*1
    train_gamma_u = True
    iterations = 0
    
    while True:
        alpha_pre = alpha
        beta_u_pre = beta_u
        beta_i_pre = beta_i
        gamma_i_pre = gamma_i
        gamma_u_pre = gamma_u
        
        alpha = 0.0
        for l in data:
            alpha += l[2]-beta_u[ind_in_beta_u[l[0]]]-beta_i[ind_in_beta_i[l[1]]]-gamma_u[ind_in_beta_u[l[0]]]-gamma_i[ind_in_beta_i[l[1]]]
        alpha /= len(data)
        
        beta_u = np.zeros(len(users))
        for l in data:
            beta_u[ind_in_beta_u[l[0]]] += l[2]-alpha-beta_i[ind_in_beta_i[l[1]]]-gamma_u[ind_in_beta_u[l[0]]]-gamma_i[ind_in_beta_i[l[1]]]
        for i in range(len(users)):
            beta_u[i] /= (lam+count_u[users[i]])
            
        beta_i = np.zeros(len(businesses))
        for l in data:
            beta_i[ind_in_beta_i[l[1]]] += l[2]-alpha-beta_u[ind_in_beta_u[l[0]]]-gamma_u[ind_in_beta_u[l[0]]]-gamma_i[ind_in_beta_i[l[1]]]
        for i in range(len(businesses)):
            beta_i[i] /= (lam+count_i[businesses[i]])
            
        if train_gamma_u:
            gamma_u = np.zeros(len(users))
            gamma_i_sqsum = 0
            for l in data:
                gamma_u[ind_in_beta_u[l[0]]] += gamma_i[ind_in_beta_i[l[1]]]*(l[2]
                -alpha-beta_i[ind_in_beta_i[l[1]]]-beta_u[ind_in_beta_u[l[0]]])
                gamma_i_sqsum += gamma_i[ind_in_beta_i[l[1]]]*gamma_i[ind_in_beta_i[l[1]]]
            for i in range(len(users)):
                gamma_u[i] /= (lam+gamma_i_sqsum)
            d = (alpha-alpha_pre)**2+sum((beta_u-beta_u_pre)**2)/len(users)+sum((beta_i-beta_i_pre)**2)/len(businesses)+sum((gamma_u-gamma_u_pre)**2)/len(users)
            if d < 1e-5:
                train_gamma_u = False
                iterations += 1
        
        if not train_gamma_u:
            gamma_i = np.zeros(len(businesses))
            gamma_u_sqsum = 0
            for l in data:
                gamma_i[ind_in_beta_i[l[1]]] += gamma_u[ind_in_beta_u[l[0]]]*(l[2]-alpha-beta_i[ind_in_beta_i[l[1]]]-beta_u[ind_in_beta_u[l[0]]])
                gamma_u_sqsum += gamma_u[ind_in_beta_u[l[0]]]*gamma_u[ind_in_beta_u[l[0]]]
            for i in range(len(users)):
                gamma_i[i] /= (lam+gamma_u_sqsum)
            d = (alpha-alpha_pre)**2+sum((beta_u-beta_u_pre)**2)/len(users)+sum((beta_i-beta_i_pre)**2)/len(businesses)+sum((gamma_i-gamma_i_pre)**2)/len(businesses)
            if d < 1e-5:
                train_gamma_u = True
                iterations += 1
                print iterations
        
        if iterations > 10:
            break
        
    return alpha, beta_u, beta_i, gamma_u, gamma_i

def squaredDiff(x, y):
        sum = 0.0
        for a,b in zip(x,y):
            sum += 1.0*(a-b)*(a-b)
        return sum

def readingFile(filename):
    f = open(filename, "r")
    data = []
    for row in f:
        if row.startswith("userId"):
            continue
        r = row.split(',')
        e = [int(r[0]), int(r[1]), float(r[2])]
        data.append(e)
    return data

train = readingFile("rating_train.csv")
test = readingFile("rating_test.csv")

users = set()
businesses = set()
for l in train:
    users.add(l[0])
    businesses.add(l[1])

users = list(users)
businesses = list(businesses)

ind_in_beta_u = defaultdict(int)
for i in range(len(users)):
    ind_in_beta_u[users[i]] = i
ind_in_beta_i = defaultdict(int)
for i in range(len(businesses)):
    ind_in_beta_i[businesses[i]] = i
    
count_u = defaultdict(int)
count_i = defaultdict(int)
for l in train:
    count_u[l[0]] += 1
    count_i[l[1]] += 1

alpha, beta_u, beta_i, gamma_u, gamma_i = train_para(train, 4)
predictions=[alpha+beta_u[ind_in_beta_u[l[0]]]+beta_i[ind_in_beta_i[l[1]]]+gamma_u[ind_in_beta_u[u]]*gamma_i[ind_in_beta_i[i]] for l in test]
y_val = [l[2] for l in test]
mse = squaredDiff(predictions, y_val) / len(y_val)
print("lfm2 mse:"+str(mse))