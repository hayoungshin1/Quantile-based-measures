np.random.seed(50)
m=12
numlevels=3
n=2
N=300
B=400
ext=0.98
dispersion=np.zeros((B,4*numlevels))
skewness=np.zeros((B,4*numlevels))
kurtosis=np.zeros((B,4*numlevels))
mdispersion=np.zeros((B,numlevels))
mskewness=np.zeros((B,numlevels))
mkurtosis=np.zeros((B,numlevels))
supdispersion=np.zeros((B,2*numlevels))
avedispersion=np.zeros((B,2*numlevels))
supskewness=np.zeros((B,2*numlevels))
aveskewness=np.zeros((B,2*numlevels))
supkurtosis=np.zeros((B,2*numlevels))
avekurtosis=np.zeros((B,2*numlevels))
sasymmetry=np.zeros((B,2*numlevels))

for l in range(B):
    for moment in ['dispersion', 'skewness', 'kurtosis', 'sasymmetry']:
        for extreme in range(2):
            for level in range(numlevels):
                if moment=='dispersion':
                    X=np.random.normal(0,1,(2,N))/2
                    X[1,:]*=2/(2**level)
                if moment=='skewness':
                    X=skewnorm.rvs(2*(numlevels-level-1)**2,size=(2,N))
                    X[1,:]/=2
                if moment=='kurtosis':
                    X=t.rvs(2*level**2+5,size=(2,N))
                    X[1,:]/=2
                if moment=='sasymmetry':
                    X=skewnorm.rvs(2*(numlevels-level-1)**2,size=(2,N))
                if extreme==0:
                    mX=np.mean(X,1,keepdims=True)
                    Sinv=np.linalg.inv(np.cov(X))
                    if moment=='dispersion':
                        betas=np.array([0.5])
                        for k in range(2):
                            dispersion[l,2*numlevels*extreme+2*level+k]=np.quantile(X[k,:],0.5+betas[0]/2)-np.quantile(X[k,:],0.5-betas[0]/2)
                        mdispersion[l,level]=np.sum((X-mX)**2)/N
                    elif moment=='skewness':
                        betas=np.array([0.5])
                        for k in range(2):
                            skewness[l,2*numlevels*extreme+2*level+k]=(np.quantile(X[k,:],0.5+betas[0]/2)+np.quantile(X[k,:],0.5-betas[0]/2)-2*np.median(X[k,:]))/(np.quantile(X[k,:],0.5+betas[0]/2)-np.quantile(X[k,:],0.5-betas[0]/2))
                        mskewness[l,level]=np.abs(np.sum(np.matmul(np.matmul(np.transpose(X-mX),Sinv),X-mX)**3)/(N**2))
                    elif moment=='kurtosis':
                        betas=np.array([0.2,0.8])
                        for k in range(2):
                            kurtosis[l,2*numlevels*extreme+2*level+k]=(np.quantile(X[k,:],0.5+betas[1]/2)-np.quantile(X[k,:],0.5-betas[1]/2))/(np.quantile(X[k,:],0.5+betas[0]/2)-np.quantile(X[k,:],0.5-betas[0]/2))
                        mkurtosis[l,level]=np.trace(np.matmul(np.matmul(np.transpose(X-mX),Sinv),X-mX)**2)/N
                    elif moment=='sasymmetry':
                        betas=np.array([0.5])
                if extreme==1:
                    if moment=='dispersion':
                        betas=np.array([ext])
                        for k in range(2):
                            dispersion[l,2*numlevels*extreme+2*level+k]=np.quantile(X[k,:],0.5+betas[0]/2)-np.quantile(X[k,:],0.5-betas[0]/2)
                    elif moment=='skewness':
                        betas=np.array([ext])
                        for k in range(2):
                            skewness[l,2*numlevels*extreme+2*level+k]=(np.quantile(X[k,:],0.5+betas[0]/2)+np.quantile(X[k,:],0.5-betas[0]/2)-2*np.median(X[k,:]))/(np.quantile(X[k,:],0.5+betas[0]/2)-np.quantile(X[k,:],0.5-betas[0]/2))
                    elif moment=='kurtosis':
                        betas=np.array([0.2,ext])
                        for k in range(2):
                            kurtosis[l,2*numlevels*extreme+2*level+k]=(np.quantile(X[k,:],0.5+ext/2)-np.quantile(X[k,:],0.5-ext/2))/(np.quantile(X[k,:],0.5+0.2/2)-np.quantile(X[k,:],0.5-0.2/2))
                    elif moment=='sasymmetry':
                        betas=np.array([ext])
                u=np.array([[0,0]])
                for i in range(len(betas)):
                    for j in range(2*m):
                        u=np.concatenate((u,betas[i]*np.array([[np.cos(j*2*math.pi/(2*m)),np.sin(j*2*math.pi/(2*m))]])),axis=0)
                quantiles=quantile(u,X)
                logs=quantiles[1:,:]-np.expand_dims(quantiles[0,:],0)
                pluslogs=logs[0:m,:]+logs[m:2*m,:]
                minuslogs=logs[0:m,:]-logs[m:2*m,:]
                supinterrange=np.max(np.sqrt(np.sum(minuslogs**2,1)))
                aveinterrange=np.mean(np.sqrt(np.sum(minuslogs**2,1)))
                if moment=='dispersion':
                    supdispersion[l,numlevels*extreme+level]=supinterrange
                    avedispersion[l,numlevels*extreme+level]=aveinterrange
                elif moment=='skewness':
                    supskewness[l,numlevels*extreme+level]=np.max(np.sqrt(np.sum(pluslogs**2,1)))/supinterrange
                    vector=np.mean(pluslogs,0)/aveinterrange
                    aveskewness[l,numlevels*extreme+level]=np.sum(vector**2)
                    sasymmetry[l,numlevels*extreme+level]=np.log(np.sqrt(np.max(np.sum(logs**2,1))/np.min(np.sum(logs**2,1))))
                elif moment=='kurtosis':
                    minuslogs=logs[2*m:3*m,:]-logs[3*m:4*m,:]
                    supkurtosis[l,numlevels*extreme+level]=np.max(np.sqrt(np.sum(minuslogs**2,1)))/supinterrange
                    avekurtosis[l,numlevels*extreme+level]=np.mean(np.sqrt(np.sum(minuslogs**2,1)))/aveinterrange
                elif moment=='sasymmetry':
                    sasymmetry[l,numlevels*extreme+level]=np.log(np.sqrt(np.max(np.sum(logs**2,1))/np.min(np.sum(logs**2,1))))
    print(l)


if True:
    print('dispersion mean:')
    print(np.mean(dispersion,0))
    print('skewness mean:')
    print(np.mean(skewness,0))
    print('kurtosis mean:')
    print(np.mean(kurtosis,0))
    print('mdispersion mean:')
    print(np.mean(mdispersion,0))
    print('mskewness mean:')
    print(np.mean(mskewness,0))
    print('mkurtosis mean:')
    print(np.mean(mkurtosis,0))
    print('supdispersion mean:')
    print(np.mean(supdispersion,0))
    print('avedispersion mean:')
    print(np.mean(avedispersion,0))
    print('supskewness mean:')
    print(np.mean(supskewness,0))
    print('aveskewness mean:')
    print(np.mean(aveskewness,0))
    print('supkurtosis mean:')
    print(np.mean(supkurtosis,0))
    print('avekurtosis mean:')
    print(np.mean(avekurtosis,0))
    print('sasymmetry mean:')
    print(np.mean(sasymmetry,0))

if True:
    print('dispersion standard error:')
    print(np.std(dispersion,0)/np.sqrt(B))
    print('skewness standard error:')
    print(np.std(skewness,0)/np.sqrt(B))
    print('kurtosis standard error:')
    print(np.std(kurtosis,0)/np.sqrt(B))
    print('mdispersion standard error:')
    print(np.std(mdispersion,0)/np.sqrt(B))
    print('mskewness standard error:')
    print(np.std(mskewness,0)/np.sqrt(B))
    print('mkurtosis standard error:')
    print(np.std(mkurtosis,0)/np.sqrt(B))
    print('supdispersion standard error:')
    print(np.std(supdispersion,0)/np.sqrt(B))
    print('avedispersion standard error:')
    print(np.std(avedispersion,0)/np.sqrt(B))
    print('supskewness standard error:')
    print(np.std(supskewness,0)/np.sqrt(B))
    print('aveskewness standard error:')
    print(np.std(aveskewness,0)/np.sqrt(B))
    print('supkurtosis standard error:')
    print(np.std(supkurtosis,0)/np.sqrt(B))
    print('avekurtosis standard error:')
    print(np.std(avekurtosis,0)/np.sqrt(B))
    print('sasymmetry standard error:')
    print(np.std(sasymmetry,0)/np.sqrt(B))

###### visualization

colors=['tab:orange','tab:green','tab:red']
m=12
N=30000

for moment in ['dispersion', 'skewness', 'kurtosis', 'sasymmetry']:
    for extreme in range(2):
        f = plt.figure(figsize=(7,7))
        ax = plt.gca()
        plt.scatter(0, 0, s=30, c='tab:blue', marker = '.')
        for level in range(numlevels):
            np.random.seed(1)
            if moment=='dispersion':
                X=np.random.normal(0,1,(2,N))/2
                X[1,:]*=2/(2**level)
            if moment=='skewness':
                X=skewnorm.rvs(2*(numlevels-level-1)**2,size=(2,N))
                X[1,:]/=2
            if moment=='kurtosis':
                X=t.rvs(2*level**2+5,size=(2,N))
                X[1,:]/=2
            if moment=='sasymmetry':
                X=skewnorm.rvs(2*(numlevels-level-1)**2,size=(2,N))
            X-=np.expand_dims(np.squeeze(quantile(np.array([[0,0]]),X)),1)
            if extreme==0:
                if moment=='kurtosis':
                    betas=np.array([0.2,0.8])
                else:
                    betas=np.array([0.5])
            if extreme==1:
                if moment=='kurtosis':
                    betas=np.array([0.2,ext])
                else:
                    betas=np.array([ext])
            u=np.array([[0,0]])
            for i in range(len(betas)):
                for j in range(2*m):
                    u=np.concatenate((u,betas[i]*np.array([[np.cos(j*2*math.pi/(2*m)),np.sin(j*2*math.pi/(2*m))]])),axis=0)
            quantiles=quantile(u,X)
            color=colors[level]
            for i in range(len(betas)):
                plt.scatter(quantiles[(i*2*m+1):((i+1)*2*m+1),0], quantiles[(i*2*m+1):((i+1)*2*m+1),1], s=30, c=color, marker = '.')
                plt.plot(np.append(quantiles[(i*2*m+1):((i+1)*2*m+1),0],quantiles[(i*2*m+1):((i+1)*2*m+1),0][0])
            , np.append(quantiles[(i*2*m+1):((i+1)*2*m+1),1],quantiles[(i*2*m+1):((i+1)*2*m+1),1][0])
            , c=color)
        plt.axis('equal')
        plt.show(block=False)
