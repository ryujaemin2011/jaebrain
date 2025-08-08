from muselsl import record
import time

# 저장할 파일 경로 지정
output_file = 'EEG_recording.csv'

# EEG 데이터 기록 시작
for i in range(10):
    record(duration=1)