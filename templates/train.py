import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

names = ['preg','plas','gres','skin','test','mass','pedi','age','class']

df = pd.read_csv(url, names = names)

print(df)

array = df.values

X = array[:,0:8]
y = array[:,8]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.2, random_state = 42)

# fit the model

model = LogisticRegression()

model.fit(X_train,y_train)

# Model Accuracy

result = model.score(X_test, y_test)

print(f'The accuracy of the model is {result}')

# Save Model

joblib.dump(model, 'diabetic_75.pkl')
