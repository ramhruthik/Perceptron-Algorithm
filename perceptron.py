import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from matplotlib.pylab import rcParams
rcParams["figure.figsize"] = (14, 7)

train_data = pd.read_csv("training_data.csv", sep=" ", header=None)
train_data.columns = ['x', 'y', 'label']
display(train_data.head())

test_data = pd.read_csv("testing_data.csv", sep=" ", header=None)
test_data.columns = ['x', 'y', 'label']
display(test_data.head())

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

model = Perceptron(max_iter=800)
model.fit(train_data.drop("label", axis=1), train_data.label)

test_data["pred"] = model.predict(test_data.drop("label", axis=1))
accuracy_rate = round(accuracy_score(test_data.label, test_data.pred), 2)*100

weights = model.coef_
bias = model.intercept_

for i in range(len(test_data)):
    print(f"[{test_data.loc[i, 'x']} {test_data.loc[i, 'y']}] Actual_Label: {test_data.loc[i, 'label']} Predicted_Label: {test_data.loc[i, 'pred']}")

print(f"\n\nAccuracy Rate: {accuracy_rate}%")
print(f"\nLearned Weights are {weights}")
print(f"Learned Bias: {bias}")

plt.scatter(train_data[train_data.label == 1]['x'], train_data[train_data.label == 1]['y'], color="red", label="+ve Class")
plt.scatter(train_data[train_data.label == -1]['x'], train_data[train_data.label == -1]['y'], color="green", label="-ve Class")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Training Data")
plt.legend()
plt.show()

plt.scatter(test_data[test_data.label == 1]['x'], test_data[test_data.label == 1]['y'], color="red", label="+ve Class")
plt.scatter(test_data[test_data.label == -1]['x'], test_data[test_data.label == -1]['y'], color="green", label="-ve Class")
sns.regplot(x='x', y='label', data=test_data, label="Hyperplance")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Testing Data")
plt.legend()
plt.show()
