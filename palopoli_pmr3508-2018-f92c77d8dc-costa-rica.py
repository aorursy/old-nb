import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
house = pd.read_csv("../input/costa-rican-household-poverty-prediction/train.csv", na_values="?")
house.shape # dimensões da tabela
house.head() 
# Quali em Quanti
from sklearn import preprocessing
house = house.apply(preprocessing.LabelEncoder().fit_transform)
house.isnull().sum()
house['Target'].value_counts().plot(kind="bar") # grafico da distribuicao decrescente em barras
import numpy as np
print(house.corr())
# Treino
# variaveis explicativas
Xhouse=house[[

# 'v2a1'
 'hacdor'
, 'rooms'
, 'hacapo'
#, 'v14a'
, 'refrig'
, 'v18q'
, 'v18q1'
, 'r4h1'
, 'r4h2'
, 'r4h3'
, 'r4m1'
, 'r4m2'
, 'r4m3'
, 'r4t1'
, 'r4t2'
, 'r4t3'
, 'tamhog'
, 'tamviv'
, 'escolari'
, 'rez_esc'
, 'hhsize'
, 'paredblolad'
, 'paredzocalo'
, 'paredpreb'
, 'pareddes'
, 'paredmad'
, 'paredzinc'
, 'paredfibras'
, 'paredother'
, 'pisomoscer'
, 'pisocemento'
, 'pisoother'
, 'pisonatur'
, 'pisonotiene'
, 'pisomadera'
, 'techozinc'
, 'techoentrepiso'
, 'techocane'
, 'techootro'
, 'cielorazo'
, 'abastaguadentro'
, 'abastaguafuera'
, 'abastaguano'
, 'public'
, 'planpri'
, 'noelec'
, 'coopele'
, 'sanitario1'
, 'sanitario2'
, 'sanitario3'
, 'sanitario5'
, 'sanitario6'
, 'energcocinar1'
, 'energcocinar2'
, 'energcocinar3'
, 'energcocinar4'
, 'elimbasu1'
, 'elimbasu2'
, 'elimbasu3'
, 'elimbasu4'
, 'elimbasu5'
, 'elimbasu6'
, 'epared1'
, 'epared2'
, 'epared3'
, 'etecho1'
, 'etecho2'
, 'etecho3'
, 'eviv1'
, 'eviv2'
, 'eviv3'
, 'dis'
, 'male'
, 'female'
, 'estadocivil1'
, 'estadocivil2'
, 'estadocivil3'
, 'estadocivil4'
, 'estadocivil5'
, 'estadocivil6'
, 'estadocivil7'
, 'parentesco1'
, 'parentesco2'
, 'parentesco3'
, 'parentesco4'
, 'parentesco5'
, 'parentesco6'
, 'parentesco7'
, 'parentesco8'
, 'parentesco9'
, 'parentesco10'
, 'parentesco11'
, 'parentesco12'
, 'idhogar'
, 'hogar_nin'
, 'hogar_adul'
, 'hogar_mayor'
, 'hogar_total'
, 'dependency'
, 'edjefe'
, 'edjefa'
, 'meaneduc'
, 'instlevel1'
, 'instlevel2'
, 'instlevel3'
, 'instlevel4'
, 'instlevel5'
, 'instlevel6'
, 'instlevel7'
, 'instlevel8'
, 'instlevel9'
, 'bedrooms'
, 'overcrowding'
, 'tipovivi1'
, 'tipovivi2'
, 'tipovivi3'
, 'tipovivi4'
, 'tipovivi5'
, 'computer'
, 'television'
, 'mobilephone'
, 'qmobilephone'
, 'lugar1'
, 'lugar2'
, 'lugar3'
, 'lugar4'
, 'lugar5'
, 'lugar6'
, 'area1'
, 'area2'
, 'age'
, 'SQBescolari'
, 'SQBage'
, 'SQBhogar_total'
, 'SQBedjefe'
, 'SQBhogar_nin'
, 'SQBovercrowding'
, 'SQBdependency'
, 'SQBmeaned'
, 'agesq'

]]
# variavel a ser explicada
Yhouse = house.Target
from sklearn.neighbors import KNeighborsClassifier
k = 800 # escolha do k
knn = KNeighborsClassifier(n_neighbors=k) 
from sklearn.model_selection import cross_val_score
grupos = cross_val_score(knn,Xhouse,Yhouse,cv=10) # validacao cruzada, geracao de vetor com grupos
grupos.mean()
# a acurácia aumenta com k até que chegue em 800. Depois se mantem no mesmo valor 0.627394307911626
#Treino a base
knn.fit(Xhouse,Yhouse)
Testhouse = pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv")
num_Testhouse = Testhouse.apply(preprocessing.LabelEncoder().fit_transform)
num_Testhouse.shape
# Prection
Xtesthouse = num_Testhouse[[
# 'v2a1'
 'hacdor'
, 'rooms'
, 'hacapo'
#, 'v14a'
, 'refrig'
, 'v18q'
, 'v18q1'
, 'r4h1'
, 'r4h2'
, 'r4h3'
, 'r4m1'
, 'r4m2'
, 'r4m3'
, 'r4t1'
, 'r4t2'
, 'r4t3'
, 'tamhog'
, 'tamviv'
, 'escolari'
, 'rez_esc'
, 'hhsize'
, 'paredblolad'
, 'paredzocalo'
, 'paredpreb'
, 'pareddes'
, 'paredmad'
, 'paredzinc'
, 'paredfibras'
, 'paredother'
, 'pisomoscer'
, 'pisocemento'
, 'pisoother'
, 'pisonatur'
, 'pisonotiene'
, 'pisomadera'
, 'techozinc'
, 'techoentrepiso'
, 'techocane'
, 'techootro'
, 'cielorazo'
, 'abastaguadentro'
, 'abastaguafuera'
, 'abastaguano'
, 'public'
, 'planpri'
, 'noelec'
, 'coopele'
, 'sanitario1'
, 'sanitario2'
, 'sanitario3'
, 'sanitario5'
, 'sanitario6'
, 'energcocinar1'
, 'energcocinar2'
, 'energcocinar3'
, 'energcocinar4'
, 'elimbasu1'
, 'elimbasu2'
, 'elimbasu3'
, 'elimbasu4'
, 'elimbasu5'
, 'elimbasu6'
, 'epared1'
, 'epared2'
, 'epared3'
, 'etecho1'
, 'etecho2'
, 'etecho3'
, 'eviv1'
, 'eviv2'
, 'eviv3'
, 'dis'
, 'male'
, 'female'
, 'estadocivil1'
, 'estadocivil2'
, 'estadocivil3'
, 'estadocivil4'
, 'estadocivil5'
, 'estadocivil6'
, 'estadocivil7'
, 'parentesco1'
, 'parentesco2'
, 'parentesco3'
, 'parentesco4'
, 'parentesco5'
, 'parentesco6'
, 'parentesco7'
, 'parentesco8'
, 'parentesco9'
, 'parentesco10'
, 'parentesco11'
, 'parentesco12'
, 'idhogar'
, 'hogar_nin'
, 'hogar_adul'
, 'hogar_mayor'
, 'hogar_total'
, 'dependency'
, 'edjefe'
, 'edjefa'
, 'meaneduc'
, 'instlevel1'
, 'instlevel2'
, 'instlevel3'
, 'instlevel4'
, 'instlevel5'
, 'instlevel6'
, 'instlevel7'
, 'instlevel8'
, 'instlevel9'
, 'bedrooms'
, 'overcrowding'
, 'tipovivi1'
, 'tipovivi2'
, 'tipovivi3'
, 'tipovivi4'
, 'tipovivi5'
, 'computer'
, 'television'
, 'mobilephone'
, 'qmobilephone'
, 'lugar1'
, 'lugar2'
, 'lugar3'
, 'lugar4'
, 'lugar5'
, 'lugar6'
, 'area1'
, 'area2'
, 'age'
, 'SQBescolari'
, 'SQBage'
, 'SQBhogar_total'
, 'SQBedjefe'
, 'SQBhogar_nin'
, 'SQBovercrowding'
, 'SQBdependency'
, 'SQBmeaned'
, 'agesq'

                                    ]]
YtestPred = knn.predict(Xtesthouse)
YtestPred.shape
Id = num_Testhouse['Id']
import pandas as pd
d = {'Id' : Id, 'Target' : YtestPred}
my_df = pd.DataFrame(d)
from subprocess import check_output
sub = pd.read_csv('../input/predict/PMR3508-f92c77d8dc-prediction.csv')
sub.to_csv('sample_submission.csv', index=False)