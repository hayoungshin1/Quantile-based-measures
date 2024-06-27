import numpy as np
import math
import stat
import time
import matplotlib.pyplot as plt

n=2
N=300

def check(u,X):
    M=u.shape[0]
    degenerates=-np.ones(M)
    for i in range(N):
        v=np.expand_dims(X[:,i],1)
        index=np.where(np.sum(X==v,axis=0)!=n)[0]
        diff=X[:,index]-v
        norm=np.sqrt(np.sum(diff**2,axis=0))
        Delta=np.expand_dims(np.sum(diff/np.expand_dims(norm,0), axis=1),0)+len(index)*u
        s1=np.sqrt(np.sum(Delta**2,axis=1))
        s2=(N-len(index))*(1+np.sqrt(np.sum(u**2,axis=1)))
        degenerates[np.where(s1<=s2)]=i
    return degenerates

def loss(uprime,Xprime,q):
    qprime=np.expand_dims(q,2)
    diff=Xprime-qprime
    norm=np.sqrt(np.sum(diff**2,axis=1)) # (M,N)
    inner=np.sum(diff*uprime,axis=1)
    return np.sum(norm,axis=1)+np.sum(inner,axis=1) # shape is (M,)

def quantile(u,X,tol=1e-5,most=30): # Newton's method
    M=u.shape[0]
    out=np.zeros((M,n))
    checkers=check(u,X)
    deg=(checkers!=-1)
    if np.any(deg):
        index=np.where(deg)[0]
        out[index,:]=np.transpose(X[:,checkers[index].astype(int)])
        u=np.delete(u,index,0)
    if np.any(1-deg):
        M=u.shape[0]
        init_q=np.tile(np.expand_dims(np.median(X,axis=1),0),(M,1)) # (M,n)
        current_q=init_q
        Xprime=np.expand_dims(X,0) # (1,n,N)
        uprime=np.expand_dims(u,2) # (M,n,1)
        count=0
        step=0
        while (np.any(np.sum(step**2,0)>tol) or count==0) and count<most:
            current_q+=step
            count+=1
            current_qprime=np.expand_dims(current_q,2)
            diff=Xprime-current_qprime # (M,n,N)
            norm=np.sqrt(np.sum(diff**2,axis=1)) # (M,N)
            Delta=diff/np.expand_dims(norm,1)
            pos=np.where(norm==0)
            Delta[pos[0],:,pos[1]]=0 
            Delta=np.sum(Delta,axis=2)+N*u
            norminv=1/norm
            norminv[norm==0]=0
            Phi=np.expand_dims(np.sum(norminv,axis=1),(1,2))*np.expand_dims(np.identity(n),0)-(diff*np.expand_dims(norminv**3,1))@np.transpose(diff,(0,2,1))
            step=np.linalg.inv(Phi)@np.expand_dims(Delta,2)
            step=np.squeeze(step)
    if np.any(deg):
        index=np.where(1-deg)[0]
        out[index,:]=current_q
    else:
        out=current_q
    return out
