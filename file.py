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
    datapart = data[i].split(',')
    datapart = datapart[:5]
    print(datapart)
    time.append(i)
    TF9.append(float(datapart[1]))
    AF7.append(float(datapart[2]))
    AF8.append(float(datapart[3]))
    TP10.append(float(datapart[4]))
print(time)
print(TF9)
print(AF7)
print(AF8)
print(TP10)

plt.plot(time,TF9)
plt.plot(time,AF7)
plt.plot(time,AF8)
plt.plot(time,TP10)
plt.show()