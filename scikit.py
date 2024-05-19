# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

# Sample data
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rainy', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'High', 'Strong', 'No']
]

df = pd.DataFrame(data, columns=["outlook", "temperature", "humidity", "wind speed", "tennis?"])
le = preprocessing.LabelEncoder()
for index_value in df.columns:
    df[index_value] = le.fit_transform(df[index_value])

x = df.drop("tennis?", axis="columns")
y = df["tennis?"]
train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                    test_size=.3,
                                                    random_state=1)
model = DecisionTreeClassifier()
model.fit(train_x, train_y)
predicted_y = (model.predict(test_x))
accuracy = metrics.accuracy_score(test_y, predicted_y)
print(f"{accuracy}")
from sklearn.tree import export_text
print(export_text(model, feature_names=list(x.columns)))
