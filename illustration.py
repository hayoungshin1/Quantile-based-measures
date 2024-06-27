ran=99
m=12
numlevels=4
for moment in ['dispersion', 'skewness', 'kurtosis', 'sasymmetry']:
    data=numlevels*[0]
    minx=np.zeros(numlevels)
    maxx=np.zeros(numlevels)
    for level in range(numlevels):
        np.random.seed(ran)
        temp=np.random.normal(0,1,(2,N))/2
        if moment=='dispersion':
            X=temp
            X[1,:]*=4/(2**level)
        if moment=='skewness':
            X=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*temp**2
            X[1,:]/=2
        if moment=='kurtosis':
            X=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*(temp**3)
            X[1,:]/=2
        if moment=='sasymmetry':
            X=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*(temp**3)
        X-=np.expand_dims(np.squeeze(quantile(np.array([[0,0]]),X)),1)
        data[level]=X
        minx[level]=np.min(X)
        maxx[level]=np.max(X)
    biggest=max(max(abs(maxx)), max(abs(minx)))+0.1
    for level in range(numlevels):
        f = plt.figure(figsize=(7,7))
        ax = plt.gca()
        X=data[level]
        plt.scatter(X[0,:], X[1,:], s=30, c='0', marker = '.')
        ax.set_xlim((-biggest,biggest))
        ax.set_ylim((-biggest,biggest))
        plt.show(block=False)

dispersion=[]
skewness=[]
kurtosis=[]
supdispersion=[]
avedispersion=[]
supskewness=[]
aveskewness=[]
supkurtosis=[]
avekurtosis=[]
sasymmetry=[]

colors=['tab:orange','tab:green','tab:red','tab:purple']

for moment in ['dispersion', 'skewness', 'kurtosis', 'sasymmetry']:
    for extreme in ['no', 'yes']:
        f = plt.figure(figsize=(7,7))
        ax = plt.gca()
        plt.scatter(0, 0, s=30, c='tab:blue', marker = '.')
        for level in range(numlevels):
            np.random.seed(ran)
            temp=np.random.normal(0,1,(2,N))/2
            if moment=='dispersion':
                X=temp
                X[1,:]*=4/(2**level)
            if moment=='skewness':
                X=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*temp**2
                X[1,:]/=2
            if moment=='kurtosis':
                X=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*(temp**3)
                X[1,:]/=2
            if moment=='sasymmetry':
                X=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*(temp**3)
            X-=np.expand_dims(np.squeeze(quantile(np.array([[0,0]]),X)),1)
            if extreme=='no':
                if moment=='dispersion':
                    betas=np.array([0.5])
                    for k in range(2):
                        dispersion.append(np.quantile(X[k,:],0.5+betas[0]/2)-np.quantile(X[k,:],0.5-betas[0]/2))
                elif moment=='skewness':
                    betas=np.array([0.5])
                    for k in range(2):
                        skewness.append((np.quantile(X[k,:],0.5+betas[0]/2)+np.quantile(X[k,:],0.5-betas[0]/2)-2*np.median(X[k,:]))/(np.quantile(X[k,:],0.5+betas[0]/2)-np.quantile(X[k,:],0.5-betas[0]/2)))
                elif moment=='kurtosis':
                    betas=np.array([0.2,0.8])
                    for k in range(2):
                        kurtosis.append((np.quantile(X[k,:],0.5+betas[1]/2)-np.quantile(X[k,:],0.5-betas[1]/2))/(np.quantile(X[k,:],0.5+betas[0]/2)-np.quantile(X[k,:],0.5-betas[0]/2)))
                elif moment=='sasymmetry':
                    betas=np.array([0.5])
            if extreme=='yes':
                if moment=='dispersion':
                    betas=np.array([0.98])
                    for k in range(2):
                        dispersion.append(np.quantile(X[k,:],0.5+betas[0]/2)-np.quantile(X[k,:],0.5-betas[0]/2))
                elif moment=='skewness':
                    betas=np.array([0.98])
                    for k in range(2):
                        skewness.append((np.quantile(X[k,:],0.5+betas[0]/2)+np.quantile(X[k,:],0.5-betas[0]/2)-2*np.median(X[k,:]))/(np.quantile(X[k,:],0.5+betas[0]/2)-np.quantile(X[k,:],0.5-betas[0]/2)))
                elif moment=='kurtosis':
                    betas=np.array([0.2,0.98])
                    for k in range(2):
                        kurtosis.append((np.quantile(X[k,:],0.5+0.98/2)-np.quantile(X[k,:],0.5-0.98/2))/(np.quantile(X[k,:],0.5+0.2/2)-np.quantile(X[k,:],0.5-0.2/2)))
                elif moment=='sasymmetry':
                    betas=np.array([0.98])
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
                supdispersion.append(supinterrange)
                avedispersion.append(aveinterrange)
            elif moment=='skewness':
                supskewness.append(np.max(np.sqrt(np.sum(pluslogs**2,1)))/supinterrange)
                #aveskewness.append(np.mean(pluslogs,0)/aveinterrange)
                vector=np.mean(pluslogs,0)/aveinterrange
                aveskewness.append(np.sum(vector**2))
            elif moment=='kurtosis':
                minuslogs=logs[2*m:3*m,:]-logs[3*m:4*m,:]
                supkurtosis.append(np.max(np.sqrt(np.sum(minuslogs**2,1)))/supinterrange)
                avekurtosis.append(np.mean(np.sqrt(np.sum(minuslogs**2,1)))/aveinterrange)
            elif moment=='sasymmetry':
                sasymmetry.append(np.log(np.sqrt(np.max(np.sum(logs**2,1))/np.min(np.sum(logs**2,1)))))
            color=colors[level]
            for i in range(len(betas)):
                plt.scatter(quantiles[(i*2*m+1):((i+1)*2*m+1),0], quantiles[(i*2*m+1):((i+1)*2*m+1),1], s=30, c=color, marker = '.')
                plt.plot(np.append(quantiles[(i*2*m+1):((i+1)*2*m+1),0],quantiles[(i*2*m+1):((i+1)*2*m+1),0][0])
            , np.append(quantiles[(i*2*m+1):((i+1)*2*m+1),1],quantiles[(i*2*m+1):((i+1)*2*m+1),1][0])
            , c=color)
        plt.axis('equal')
        plt.show(block=False)

if True:
    print('dispersion:')
    print(dispersion)
    print('skewness:')
    print(skewness)
    print('kurtosis:')
    print(kurtosis)
    print('supdispersion:')
    print(supdispersion)
    print('avedispersion:')
    print(avedispersion)
    print('supskewness:')
    print(supskewness)
    print('aveskewness:')
    print(aveskewness)
    print('supkurtosis:')
    print(supkurtosis)
    print('avekurtosis:')
    print(avekurtosis)
    print('sasymmetry:')
    print(sasymmetry)
