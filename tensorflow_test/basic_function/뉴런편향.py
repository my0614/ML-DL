x = 0
y = 1
w = tf.random.normal([1],0,1) # 가중치값
b = tf.random.normal([1],0,1) #편향값

for i in range(1000):
    output = sigmoid(x * w + 1*b)
    error = y - output
    w = w + x * 0.1 * error
    b = b + 1 * 0.1 * error 
    
    if i % 100 == 99:
        print(i, error, output) # 에러는 0에 가까워지고 output은 1에 가까워진다.