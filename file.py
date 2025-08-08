import matplotlib.pyplot as plt

f = open('EEG_recording.csv', 'r')

f.seek(0)
data = f.readlines()
data = data[1:]
f.close()

print(data)

time = []
TF9 = []
AF7 = []
AF8 = []
TP10 = []

for i in range(len(data)):
    dp = data[i].split(',')
    dp = dp[:5]
    print(dp)
    time.append(i)
    TF9.append(float(dp[1]))
    AF7.append(float(dp[2]))
    AF8.append(float(dp[3]))
    TP10.append(float(dp[4]))
print(time)
print(TF9)
print(AF7)
print(AF8)
print(TP10)

plt.plot(time,TF9,'r')
plt.plot(time,AF7,'b')
plt.plot(time,AF8,'g')
plt.plot(time,TP10,'m')
plt.show()

'''for i in time:
    td = time[:i]
    T9d = TF9[:i]
    A7d = AF7[:i]
    A8d = AF8[:i]
    T10d = TP10[:i]
    plt.plot(td, T9d,'r')
    plt.plot(td, A7d,'g')
    plt.plot(td, A8d,'b')
    plt.plot(td, T10d,'m')
    plt.pause(0.001)'''