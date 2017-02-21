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
idx = np.where((X[:,0] == 4) & (X[:,3] <= 25000) & (X[:,5] <= 200) & (X[:,5] > 100))
#idx = np.where((X[:,0] == 4) & (X[:,3] <= 35000) & (X[:,5] > 180 - 0.00054606*X[:,3]))

# --------------------
# GrossSqFt to predict MarketValueperSqFt
# --------------------
Xt=np.array(X[idx,3]).reshape(-1,1)
Yt=np.array(X[idx,5]).reshape(-1,1)
Xtmax = np.amax(Xt)
Ytmax = np.amax(Yt)
Xt = Xt/Xtmax
Yt = Yt/Ytmax
dn = 30
plt.figure(1)
plt.scatter(Xt,Yt)

# --------------------
# using a cubic polynomial
# --------------------
#deg = 15
#n,m=Xt.shape
#x = np.array(np.linspace(np.amin(Xt[:,0]),np.amax(Xt[:,0]),dn)).reshape(-1,1)
#CXt = np.zeros((n,deg))
#ct = np.zeros((dn,deg))
#for i in range(1,deg+1):
#    CXt[:,i-1]=np.power(Xt[:,0],i)
#    ct[:,i-1]=np.power(x[:,0],i)

# --------------------
# Gaussian basis
# --------------------
xmin = np.amin(Xt[:,0])
xmax = np.amax(Xt[:,0])
#center = [xmin,xmin+0.01*(xmax-xmin),xmin+0.02*(xmax-xmin),xmin+0.03*(xmax-xmin),xmin+0.04*(xmax-xmin),xmin+0.1*(xmax-xmin),xmin+0.2*(xmax-xmin),xmax]
center = [x for x in np.linspace(xmin,xmax,15)]
ncenter = len(center)
std = 0.05
n,m=Xt.shape
x = np.array(np.linspace(xmin,xmax,dn)).reshape(-1,1)
CXt = np.zeros((n,ncenter))
ct = np.zeros((dn,ncenter))
for i in range(0,ncenter):
    CXt[:,i]=np.exp((-1)*np.power((Xt[:,0]-center[i]),2)/(std*std))
    ct[:,i]=np.exp((-1)*np.power((x[:,0]-center[i]),2)/(std*std))

#for i in range(0,ncenter):
#    plt.plot(x,ct[:,i],color='red')


# --------------------
# making the linear regression
# --------------------
#CXt = Xt
#CYt = Yt
#x = np.array(np.linspace(np.amin(Xt[:,0]),np.amax(Xt[:,0]),dn)).reshape(-1,1)
#ct = x

# --------------------
# making regression
# --------------------
liregr = linear_model.LinearRegression()
bayesregrg = linear_model.BayesianRidge(alpha_1=0.1,alpha_2=0.1,lambda_1=1,lambda_2=1)
bayesregrb = linear_model.BayesianRidge(alpha_1=0.001,alpha_2=0.001,lambda_1=0.001,lambda_2=0.001)
bayesregro= linear_model.BayesianRidge(alpha_1=3,alpha_2=1,lambda_1=0.1,lambda_2=0.1)
#bayesregr = linear_model.BayesianRidge(alpha_1=1,alpha_2=1)
#bayesregr = linear_model.BayesianRidge(lambda_1=2,lambda_2=0.5)

liregr.fit(CXt, Yt)
bayesregrg.fit(CXt,Yt)
bayesregrb.fit(CXt,Yt)
bayesregro.fit(CXt,Yt)

print(liregr.coef_,liregr.intercept_)
print(bayesregrg.coef_,bayesregrg.intercept_)
print(bayesregrb.coef_,bayesregrb.intercept_)
print(bayesregro.coef_,bayesregro.intercept_)

print("MSE Linear (red): %.2f", np.mean((liregr.predict(CXt) - Yt) ** 2))
yb1 = np.array(bayesregrg.predict(CXt)).reshape(-1,1)
yb2 = np.array(bayesregrb.predict(CXt)).reshape(-1,1)
yb3 = np.array(bayesregro.predict(CXt)).reshape(-1,1)
print("MSE Bayes (green): %.2f", np.mean((yb1 - Yt) ** 2))
print("MSE Bayes (black): %.2f", np.mean((yb2 - Yt) ** 2))
print("MSE Bayes (orange): %.2f", np.mean((yb3 - Yt) ** 2))


plt.plot(x,liregr.predict(ct),color='red')
plt.plot(x,bayesregrg.predict(ct),color='green')
plt.plot(x,bayesregrb.predict(ct),color='black')
plt.plot(x,bayesregro.predict(ct),color='orange')


plt.show()
