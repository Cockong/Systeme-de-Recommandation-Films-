# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 16:11:12 2019

@author: rthie
"""

import csv
import numpy as np
import matplotlib as plt
import pylab as pl
import random
import time


def open_numeric(baseFileName, fieldnames=['user','movie','rating','datestamp'], delimiter='\t'):
    """
    fonction généralisée pour lire les fichiers numériques
    """
    with open(baseFileName, 'r') as f:
         reader = csv.DictReader(f, delimiter = delimiter, fieldnames=fieldnames)
         # create a dict out of reader, converting all values to integers
         return [dict([key, int(value)] for key, value in row.items()) for row in list(reader)]


def open_file(baseFileName, fieldnames=None , delimiter='|'):    
    """
    fonction généralisée pour lire les fichiers non numériques
    """   
    with open(baseFileName, 'r',encoding = "ISO-8859-1") as f:
         reader = csv.DictReader(f, delimiter = delimiter,fieldnames=fieldnames)
         return list(reader)

def time_elapsed(a,*args,**kwds):
    """
    retourne le temps d'exécution d'une fonction et son return
    """
    start = time.perf_counter() # first timestamp
    r=a(*args,**kwds)
    end = time.perf_counter() # second timestamp
    return (r,end-start)


def calcul_MAE(Rtest, matrice_prediction):
    """
    Calcule la MAE pour une matrice de prédiction
    :param Rtest: Matrice Test
    :param liste_matrices_prediction: liste des matrices de prédictions
    :return:la MAE
    """
    errorRating = []
    for i in range(0, Rtest.shape[0]):
        for j in range(0, Rtest.shape[1]):
            if Rtest[i][j] != 0:
                errorRating.append(matrice_prediction[i][j] - Rtest[i][j])
    return(np.mean(np.abs(errorRating)))

def resolution_equa_normale(A,b):
    """
    methode pour inverser dapres le cours
    """
    AT=A.T
    z = AT.dot(b)
    Inv = np.linalg.inv(AT.dot(A))
    return Inv.dot(z) # ne fonctionne que si A inversible cequi nest pas notre cas


baseUserItem=open_numeric("u1.base")

testUserItem=open_numeric("u1.test")



Item=open_file("u.item", fieldnames=['movieID', 'Title', 'releaseDate', 'videoReleaseDate','IMDb URL',
            'unknown','Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary','Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

User=open_file("u.user", fieldnames=['userID','age','gender','occupation','zipcode'])


NbUsers = len(User)
NbItems= len(Item)

Rtest = np.zeros((NbUsers, NbItems))
R = np.zeros((NbUsers, NbItems))
W = np.zeros((NbUsers, NbItems))
for row in baseUserItem:
    R[row['user']-1,row['movie']-1] = row['rating']
    W[row['user']-1,row['movie']-1] = 1
for row in testUserItem:
    Rtest[row['user']-1,row['movie']-1] = row['rating']
#pl.imshow(R,interpolation='none')
    
mean_items=[0 for row in range(NbItems)] #liste des moyenne par colonne(film)
for col in range(NbItems):
   rating_number=0 #savoir le nombre d'éléments quon additionne pour diviser la moyenne 
   for user_rating in R[:,col]:
       if user_rating!=0: # on ne prend que les valeurs non nul
           mean_items[col]+=user_rating
           rating_number+=1
   if rating_number!=0: #il doit y avoir au moins une note par film
       mean_items[col]=mean_items[col]/rating_number 


def prediction_moindrecarre(R, W, Rtest, nbrIter = 10 ,nbrFeatures = 1 ,lmbd = 0.01, mean_items=None):
    """    
    nbrIter =  nombre d'iterations
    nbrFeatures = k
    lmbd  = parametre lambda
    """
    score=[]
    m=R.shape[0]
    n=R.shape[1]
    X = np.ones( (m, nbrFeatures) ) # matrice initiale X
    #Y = np.ones( (nbrFeatures, n ) ) # matrice initiale Y
    #X = np.random.rand( m, nbrFeatures ) # matrice initiale X
    #Y = np.random.rand( nbrFeatures, n ) # matrice initiale Y
    #X2 = np.ones( (m, nbrFeatures) )
    
    Ik = np.eye( nbrFeatures )
    alpha= lmbd * Ik # éviter de recalculer

    if mean_items==None:
        Y = np.random.rand( nbrFeatures, n ) # matrice initiale Y  
    elif mean_items==1 :
        Y = np.ones( (nbrFeatures, n ) ) # matrice initiale Y
    else : 
        Y=np.array([mean_items for k in range(nbrFeatures)]) #approximer par la moyenne


    for j in range(nbrIter):
        
        for u in range(m):
            Wu = np.diag(W[u])
            A = Y.dot(Wu).dot(Y.T) + alpha
            b = Y.dot(Wu).dot(R[u].T)
            #attention cest X[u].T quon a calculé je ne sais i ca pose probleme
            X[u]=np.linalg.solve(A,b) # ne fonctionne que si A inversible cequi nest pas notre cas
            #X[u]=resolution_equa_normale(A,b) # ne fonctionne que si A inversible cequi nest pas notre cas
            #X[u],tempsmamethode=time_elapsed(resolution_equa_normale,A,b)
            #print(X[u])
            #X[u],tempslinalg=time_elapsed(  np.linalg.solve,A,b)#bugnp.linalg.inv(AT.dot(A))
            #print(X[u])
            #print("temps de ma méthode : ",tempsmamethode,"temps de linalg : ", tempslinalg)

    
        for i in range(n):
            Wi = np.diag(W[:,i])
            A = X.T.dot(Wi).dot(X) + alpha 
            b = X.T.dot(Wi).dot(R[:,i])
            #Y[:,i]=np.linalg.solve(A,b) # ne fonctionne que si A inversible cequi nest pas notre cas
            Y[:,i]=resolution_equa_normale(A,b) # ne fonctionne que si A inversible cequi nest pas notre cas

        
        P=X.dot(Y) #     Calculer la prediction pour chaque rating du jeu de données test
        score.append(calcul_MAE(Rtest, P))
        
    return score

#meilleurs résultats k=2 nbriter=7 (apres diverge) et simulation par 111

score=[]

for k in range(1,5):
    print("approximation 1 avec k = ", k)
    newscore1,temps=time_elapsed(prediction_moindrecarre,R,W,Rtest,mean_items=1,nbrFeatures=k)  
    print("score = {} en {} secondes".format(newscore1[9],temps))
    print()
    print("approximation moyenne avec k = ", k)
    newscore2,temps=time_elapsed(prediction_moindrecarre,R,W,Rtest,mean_items=mean_items,nbrFeatures=k)
    print("score = {} en {} secondes".format(newscore2[9],temps))
    print()
    print("approximation random avec k = ", k)
    newscore3,temps=time_elapsed(prediction_moindrecarre,R,W,Rtest,nbrFeatures=k)
    print("score = {} en {} secondes".format(newscore3[9],temps))

    score.append((newscore1,newscore2,newscore3))

"""
for l in range(1,10) :   
    lmbd=l*0.01
    print(lmbd)
    print("approximation moyenne")
    newscore1,temps=time_elapsed(prediction_moindrecarre,R,W,Rtest,lmbd = lmbd,mean_items=mean_items)
    print(temps)
    print(newscore1[9])
    print("approximation 1")
    newscore2,temps=time_elapsed(prediction_moindrecarre,R,W,Rtest,lmbd = lmbd,mean_items=1)
    print(temps)
    print(newscore2[9])
    print("approximation random")
    newscore3,temps=time_elapsed(prediction_moindrecarre,R,W,Rtest,lmbd = lmbd)
    print(temps)
    print(newscore3[9])        
    score.append((newscore1,newscore2,newscore3))


#approximation de X et Y par la moyenne
#fonctionne moins bien meilleur départ mais cnvergence moins bonne HYPOTHESE (nameliore rien)=> apparemment il ya des 0 dans Y : des films nont pas été notés !!! je vais essayer de remplacer ces zeros par des 1

mean_users=[0 for row in range(NbUsers)] #liste des moyenne par lignes(users)
for row in range(NbUsers):
   rating_number=0 #savoir le nombre d'éléments quon additionne pour diviser la moyenne
   for user_rating in R[row,:]:
       if user_rating!=0:
           mean_users[row]+=user_rating
           rating_number+=1
   if rating_number!=0:    #il doit y avoir au moins une note par film   
       mean_users[row]=mean_users[row]/rating_number      

X=np.array([mean_users for k in range(nbrFeatures)]) #approximer par la moyenne

X=X.T
 
mean_items=[0 for row in range(NbItems)] #liste des moyenne par colonne(film)
for col in range(NbItems):
   rating_number=0 #savoir le nombre d'éléments quon additionne pour diviser la moyenne 
   for user_rating in R[:,col]:
       if user_rating!=0: # on ne prend que les valeurs non nul
           mean_items[col]+=user_rating
           rating_number+=1
   if rating_number!=0: #il doit y avoir au moins une note par film
       mean_items[col]=mean_items[col]/rating_number 

Y=np.array([mean_items for k in range(nbrFeatures)]) #approximer par la moyenne

          

for j in range(nbrIter):
    
    for u in range(m):
        Wu = np.diag(W[u])
        A = Y.dot(Wu).dot(Y.T) + alpha
        b = Y.dot(Wu).dot(R[u].T)
        #attention cest X[u].T quon a calculé je ne sais i ca pose probleme
        X[u]=np.linalg.solve(A,b) # ne fonctionne que si A inversible cequi nest pas notre cas

    for i in range(n):
        Wi = np.diag(W[:,i])
        A = X.T.dot(Wi).dot(X) + alpha 
        b = X.T.dot(Wi).dot(R[:,i])
        Y[:,i]=np.linalg.solve(A,b) # ne fonctionne que si A inversible cequi nest pas notre cas
        

    
    P=X.dot(Y) #     Calculer la prediction pour chaque rating du jeu de données test
    score=calcul_MAE(Rtest, P)
    print(score)
    

    
#test comparaison des deux méthodes    
    
print(X.shape,X2.shape)
n1=X.shape[0]
n2=X.shape[1]
nerror=0
epsilon=10**(-6)
for ligne in range(n1-1):
    for coeff in range(n2-1):
        if abs(X2[n1,n2]-X[n1,n2])>epsilon:
            nerror+=1
            print("False")



#enleve les valeurs negatives de Y
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        if Y[i][j]==0:
            Y[i][j]=1


# exemple utilisation temps
            
start = time.perf_counter() # first timestamp

# we place here the code we want to time

end = time.perf_counter() # second timestamp

elapsed = end - start
print("elapsed time = {:.12f} seconds".format(elapsed))




"""   
