import matplotlib.pyplot as plt
import random

xd = []
yd = []

for i in range(1000):
    xd.append(random.randrange(1,10))
    yd.append(random.randrange(1,10))

print(xd)
print(yd)

for i in xd:
    x = xd[i-10:i]
    y = yd[i-10:i]
    plt.clf()
    plt.plot(x,y,'r')
    plt.pause(0.01)