import subprocess
import time
from pylsl import StreamInlet, resolve_byprop
import matplotlib.pyplot as plt

# 1. muselsl stream을 백그라운드에서 실행
print("Muse 스트림 프로세스 시작 중...")
stream_process = subprocess.Popen(["muselsl", "stream"])

# 2. 스트림 시작까지 기다림 (Muse 연결 시간 필요)
print("스트림 연결 대기 중...")
time.sleep(5)  # 5초 정도 대기, 필요시 조정

# 3. LSL EEG 스트림 검색
print("EEG stream 검색 중...")
streams = resolve_byprop('type', 'EEG', timeout=10)

if not streams:
    print("❌ EEG 스트림을 찾을 수 없습니다. Muse 연결을 확인하세요.")
    stream_process.terminate()
    exit(1)

# 4. LSL 스트림 수신
inlet = StreamInlet(streams[0])
print("✅ EEG stream 연결 완료!")

try:
    print("실시간 EEG 데이터 수신 중 (Ctrl+C로 종료)...")
    while True:
        sample, timestamp = inlet.pull_sample()
        print(sample[0])
        plt.plot(timestamp, sample[0])
        plt.pause(0.01)
except KeyboardInterrupt:
    print("\n데이터 수신 중지됨.")
finally:
    # 5. stream 프로세스 종료
    print("Muse 스트림 프로세스 종료 중...")
    stream_process.terminate()
