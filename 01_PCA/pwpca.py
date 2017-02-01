import csv
import numpy as np
import matplotlib.pyplot as plt
#from math import log1p
from sklearn.decomposition import PCA


#redata = open('/Users/lgnonato/Meusdocs/Cursos/CUSP-GX-5006/Data/Manhattan-RE.csv', "r")
redata = open('./manhattan-dof.csv', "r")
csvReader = csv.reader(redata,delimiter=';')
next(csvReader)

#############
# Case 1 (amount of explained variance)
#############
# X = np.array([r for r in csvReader])
# X = X.astype(np.float)

# tpca=PCA(svd_solver='full')
# tpca.fit(X)
# print(tpca.explained_variance_)
# print(tpca.explained_variance_ratio_)

# plt.figure(1)
# plt.plot(np.log(tpca.explained_variance_), '*')
# plt.show()

#############
# Case 2 (visualizing the data)
#############
# X = np.array([r for r in csvReader])
# X = X.astype(np.float)

# tpca=PCA(n_components=2)
# Xp=tpca.fit_transform(X)
# print(tpca.explained_variance_ratio_)

# plt.figure(2)
# plt.scatter(Xp[:,0],Xp[:,1])
# plt.show()

#############
# Case 3 (Exploring subsets)
#############

# X = np.array([[r[0],r[1],r[2],r[4],r[5]] for r in csvReader])
# X=X.astype(np.float)

# tpca=PCA(n_components=3)
# Xp=tpca.fit_transform(X)
# print('fig 1:', tpca.explained_variance_ratio_)

# plt.figure(1)
# plt.scatter(Xp[:,0],Xp[:,1])
# #plt.scatter(X[:,3],X[:,4])
# plt.show()

#############
# Case 4 (Exploring subsets)
#############
# redata.seek(0)
# next(csvReader)

# # YearBuilt AssessLand NumFloors
# X1 = np.array([[r[0],r[1],r[2],r[3],r[5]] for r in csvReader])
# X1 = X1.astype(np.float)

# t1pca=PCA(n_components=2)
# Xp1=t1pca.fit_transform(X1)
# print('fig 2:', t1pca.explained_variance_ratio_)

# plt.figure(2)
# #plt.scatter(Xp1[:,0],Xp1[:,1])
# plt.scatter(X1[:,3],X1[:,4])
# plt.show()
