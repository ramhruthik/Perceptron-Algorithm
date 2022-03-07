import numpy as np
import random
import matplotlib.pyplot as plt
import time

# plot y=10x+3
true_x = [random.randint(0,1000) for _ in range(20)]
true_y = list()
for i in range(len(true_x)):
    true_y.append((true_x[i]*10)+3)

#generate 20 random points x and y
x_value = []
y_value = []
for _ in range(20):
    x_value.append(random.randint(0, 1000))
    y_value.append(random.randint(0, 1000))

#assigning sign postive and negative
def classify(x, y):
    result = x-y+2
    if result >= 0:
        return 1
    else:
        return -1


data = []
total = []
labels = []
for x, y in zip(x_value, y_value):
    label = classify(x, y)
    data.append([x, y, label])
    total.append([x, y])
    labels.append(label)

#assigning signs to linearly seperated data
xA = []
yA = []
xB = []
yB = []
for i in range(len(data)):
    if data[i][-1] == 1:
        xA.append(data[i][0])
        yA.append(data[i][1])
    elif data[i][-1] == -1:
        xB.append(data[i][0])
        yB.append(data[i][1])

plt.plot(true_x, true_y, '-g')
plt.plot(xA, yA, 'o', markersize=5)
plt.plot(xB, yB, 'o', markerfacecolor='none', markersize=5)
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.title("Initial Plotted Data")
plt.show(block=False)
plt.pause(2)
plt.close()

class Perceptron:
    def __init__(self):
        self.weight = None
        self.bias = None

    def model(self, x):
        print("Weights:",self.weight,"Data:", x)
        if np.dot(self.weight, x) >= self.bias:
            return 1
        else:
            return -1

    def get_points(self):
        point_x = [random.randint(0, 1000) for _ in range(20)]
        point_y = list()
        for i in range(len(point_x)):
            point_y.append((self.weight[0]*(point_x[i])+self.bias)/abs(self.weight[1]))
        return point_x, point_y

    def fit(self, data, labels, total_data,true_x, true_y,xA,yA,xB,yB, learning_rate=1):
        # Randomly initialize the weights to a double within (0, 1).
        self.weight = np.array([random.uniform(0, 1), random.uniform(0, 1)])
        self.bias = random.uniform(0, 1)
        plt.ion()
        epochs = 1
        while epochs:
            pred_y = []
            for x in range(len(data)):
                pred = self.model(data[x])
                pred_y.append(pred)
                if labels[x] == 1 and pred <= 0:
                    self.weight = self.weight + learning_rate * np.array(data[x])
                    self.bias = self.bias + learning_rate * 1
                elif labels[x] == -1 and pred >= 1:
                    self.weight = self.weight - learning_rate * np.array(data[x])
                    self.bias = self.bias - learning_rate * 1
             #Visualize the line in green and the points (circles filled or unfilled) on a graphic
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.rcParams.update({'figure.max_open_warning': 0})
            line1, = ax.plot(true_x, true_y, '-g')
            line3, = ax.plot(xA, yA, 'o', markersize=5)
            line4, = ax.plot(xB, yB, 'o', markerfacecolor='none', markersize=5)
            points_x, points_y = self.get_points()
            line1.set_xdata(points_x)
            line1.set_ydata(points_y)
            ax.set_title(f'Epochs : {epochs}', fontsize=15)
            mis_pts = len(labels)-np.count_nonzero(np.array(pred_y) == np.array(labels))
            print("Epoch :", epochs, end=" - ")
            print("number of missclassification points :", mis_pts)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1)
            if mis_pts == 0:
                break
            epochs += 1

perceptron = Perceptron()
perceptron.fit(total,labels,data,true_x,true_y,xA,yA,xB,yB,learning_rate = 0.0001)
