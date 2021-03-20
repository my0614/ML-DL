import matplotlib.pyplot as plt
import numpy as np
X = [0.3,-0.78,1.26,0.03,1.11,0.24,-0.24,-0.47,-0.77,0.37,-0.85,-0.41,-0.27,0.02,-0.76,2.66]
Y = [12.27,14.44,11.87,18.75,17.52,16.37,19.78,19.51,12.65,14.74,10.72,21.94,12.83,15.51,17.14,14.42]

x_bar = sum(X) / len(X) #x의평균
y_bar = sum(Y) / len(Y) #y의평균

#최소제곱법 
a = sum([(y - y_bar) * (x - x_bar) for y, x in list(zip(Y,X))])
a /= sum([(x - x_bar) ** 2 for x in X])
b = y_bar - a * x_bar
print('a :', a, 'b:',b)

line_x = np.arange(min(X), max(X), 0.01)
print('line_x',line_x)
line_y = a * line_x + b # 1차함수의 직선그래프 식 y = ax + b

plt.plot(line_x, line_y, 'r-')
plt.plot(X,Y,'bo')
plt.xlabel('population growth rate (%)')
plt.ylabel('elderly growth rate (%)')
plt.show()