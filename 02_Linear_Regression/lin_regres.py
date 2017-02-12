from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import csv

redata = open('/Users/lgnonato/Meusdocs/Cursos/CUSP-GX-5006/Data/manhattan-dof.csv', "r")
csvReader = csv.reader(redata,delimiter=';')
next(csvReader)

X = np.array([r for r in csvReader])
X = X.astype(np.float)

# --------------------
# filtering data
# --------------------
#c = 0
#idx = np.where((X[:,0] == 4) & (X[:,3] <= 35000) )
idx = np.where((X[:,0] == 4) & (X[:,3] <= 35000) & (X[:,5] > 180 - 0.00054606*X[:,3]))

# --------------------
# GrossSqFt to predict MarketValueperSqFt
# --------------------
#Xt=np.array(X[idx,3]).reshape(-1,1)
#Yt=np.array(X[idx,5]).reshape(-1,1)
#Xtmax = np.amax(Xt)
#Ytmax = np.amax(Yt)
#Xt = Xt/Xtmax
#Yt = Yt/Ytmax
#plt.figure(1)
#plt.scatter(Xt,Yt)

# --------------------
# making the linear regression
# --------------------
#liregr = linear_model.LinearRegression()
#riregr = linear_model.Ridge(alpha=1.0e2)
#laregr = linear_model.Lasso(alpha=1.0e2)
#liregr.fit(Xt, Yt)
#riregr.fit(Xt, Yt)
#laregr.fit(Xt, Yt)
#print(liregr.coef_,liregr.intercept_)
#print(riregr.coef_,riregr.intercept_)
#print(laregr.coef_,riregr.intercept_)
#
#print("MSE Linear (red): %.2f", np.mean((liregr.predict(Xt) - Yt) ** 2))
#print("MSE Ridge (green): %.2f", np.mean((riregr.predict(Xt) - Yt) ** 2))
#print("MSE Lasso (black): %.2f", np.mean((laregr.predict(Xt) - Yt) ** 2))
#plt.scatter(Xt,liregr.predict(Xt),color='red')
#plt.scatter(Xt,riregr.predict(Xt),color='green')
#plt.scatter(Xt,laregr.predict(Xt),color='black')


# --------------------
# several variables to predict MarketValueperSqFt
# --------------------
concat = [0,2,3,4]
t = []
for i in concat:
    t.append(X[idx,i])
Xt=np.array(t)
Xt=Xt.reshape((-1,len(concat)))
Yt=np.array(X[idx,5]).reshape(-1,1)
Xtmax = np.amax(Xt,axis=0)
for i in range(0,len(concat)):
    Xt[:,i]=Xt[:,i]/Xtmax[i]
Ytmax = np.amax(Yt)
Yt = Yt/Ytmax
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(Xt[:,0], Xt[:,1], Yt)
#ax.scatter(Xt[:,0], Xt[:,1], liregr.predict(Xt),color='red')
#ax.scatter(Xt[:,0], Xt[:,1], riregr.predict(Xt),color='green')
#ax.scatter(Xt[:,0], Xt[:,1], laregr.predict(Xt),color='black')

# --------------------
# exploring regularization values
# --------------------
a =  np.linspace(0.01,10,50)
eri = []
ela = []
pari = []
pala = []
for i in a:
    riregr = linear_model.Ridge(alpha=i)
    riregr.fit(Xt, Yt)
    eri.append(np.mean((riregr.predict(Xt) - Yt) ** 2)) # MSE ridge
    pari.append(riregr.coef_)

    laregr = linear_model.Lasso(alpha=i)
    laregr.fit(Xt, Yt)
    ela.append(np.mean((laregr.predict(Xt) - Yt) ** 2)) # MSE lasso
    pala.append(laregr.coef_)

plt.figure(1)
plt.plot(a,eri,color='red')
plt.plot(a,ela,color='green')

plt.figure(2)
plt.plot([i[0] for i in pari],color='red')
#plt.plot([i[0] for i in pala],color='green')

print("Ridge min: ", min(eri))
print("Lasso min: ", min(ela))


# --------------------
# distribution per neighborhood / year
# --------------------
#plt.figure(2)
#dnbr = dict((x,0) for x in X[:,c])
#for x in X[:,c]:
#    dnbr[x]+=1
#
#for i in dnbr:
#    plt.scatter(i,dnbr[i],color='blue')

plt.show()
