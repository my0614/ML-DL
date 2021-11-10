train_20 = [37.8,8.2,5.6,31.3,9.5,3.0,4.2]
train_21 = [42.8,8.6,4.4,32.5,3.6,3.6,4.6]

all_data = 512*512*142
q_data = 512*512*71
print('all_data', all_data)
for c in range(7):
    val = int(q_data * train_20[c]/100)
    val2 = int(q_data * train_21[c]/100)
    print('class%d ' % (c+1),(val+ val2) / all_data * 100)
