import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

df=pd.read_csv('final_data.csv')
df=df.drop(columns=['player', 'team', 'name','position'])

X = df.drop(columns=["current_value"])
y=df['current_value']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=11)

sc=StandardScaler()
X_train.iloc[:,:]=sc.fit_transform(X_train.iloc[:,:])
X_test.iloc[:,:]=sc.transform(X_test.iloc[:,:])

dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
y_pred=dtr.predict(X_test)

print("r2_score:",r2_score(y_test,y_pred))


# # new_player = [180, 22, 34, 0, 0.033507074, 0.335070737, 0, 0, 0, 0, 2686, 175, 28, 1, 12000000, 2, 1]
# new_player=[194.000000,31.0,61,0.000000,0.000000,0.064366,0.0,0.000000,0.917218,0.386197,5593,183,25,5,32000000,1,0]
# # new_player=[1.834496,1.039337,0.921085,-0.518867,-0.598732,-0.284760,-0.179164,-0.079358,1.841253,0.334180,1.543761,0.378609,0.399002,0.806601,1.910726,-1.743904,-0.666471]
# new_player_array = np.array(new_player)
# new_player_2darray= new_player_array.reshape(1, -1)
# print(dtr.predict(new_player_2darray))
