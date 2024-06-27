np.random.seed(1)

coverage=np.zeros(2)

betas=np.array([0.5,0.98])
m=12
u=np.array([[0,0]])
for i in range(len(betas)):
    for j in range(2*m):
        u=np.concatenate((u,betas[i]*np.array([[np.cos(j*2*math.pi/(2*m)),np.sin(j*2*math.pi/(2*m))]])),axis=0)


T=1000
reps=1000
stotal=len(betas)*[0]
for k in range(reps):
    skewness=len(betas)*[0]
    bootstrapped_skewness=len(betas)*[0]
    r=len(betas)*[0]
    X=np.random.normal(size=(n,N))
    quantiles=quantile(u,X,tol=1e-10,most=50)
    logs=quantiles[1:,:]-np.expand_dims(quantiles[0,:],0)
    for i in range(len(betas)):
        pluslogs=logs[i*2*m:(i*2+1)*m,:]+logs[(i*2+1)*m:(i*2+2)*m,:]
        minuslogs=logs[i*2*m:(i*2+1)*m,:]-logs[(i*2+1)*m:(i*2+2)*m,:]
        skewness[i]=np.mean(pluslogs,0)/np.mean(np.sqrt(np.sum(minuslogs**2,1)))
        bootstrapped_skewness[i]=np.zeros((T,n))
    for t in range(T):
        quantiles=quantile(u,X[:,np.random.choice(N,size=N,replace=True)],tol=1e-10,most=50)
        logs=quantiles[1:,:]-np.expand_dims(quantiles[0,:],0)
        for i in range(len(betas)):
            pluslogs=logs[i*2*m:(i*2+1)*m,:]+logs[(i*2+1)*m:(i*2+2)*m,:]
            minuslogs=logs[i*2*m:(i*2+1)*m,:]-logs[(i*2+1)*m:(i*2+2)*m,:]
            bootstrapped_skewness[i][t,:]=np.mean(pluslogs,0)/np.mean(np.sqrt(np.sum(minuslogs**2,1)))
    for i in range(len(betas)):
        r[i]=np.sort(np.sqrt(np.sum((np.expand_dims(skewness[i],0)-bootstrapped_skewness[i])**2,1)))[int(0.95*T)]
        stotal[i]+=(np.sqrt(np.sum(skewness[i]**2))>=r[i])
    print(k, stotal)

for i in range(len(betas)):
    print(stotal[i]/(k+1))
