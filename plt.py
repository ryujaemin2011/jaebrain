import matplotlib.pyplot as plt
import numpy as np

# 데이터 초기화
x = np.arange(0, 100)
y = np.random.rand(100)
print(x)

plt.ion()  # 실시간 모드 활성화
fig, ax = plt.subplots()

line, = ax.plot(x, y)

for _ in range(100):
    y = np.random.rand(100)  # 새로운 데이터
    line.set_ydata(y)  # 데이터 갱신
    plt.draw()  # 그래프 다시 그리기
    plt.pause(0.1)  # 잠시 기다리기

plt.ioff()  # 실시간 모드 비활성화
plt.show()