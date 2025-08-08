import subprocess
import time
import signal
import sys
import os
import csv
import numpy as np
import pandas as pd
from collections import deque
from pylsl import StreamInlet, resolve_byprop
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------
# ⚙️ 설정
DATA_FOLDER = "eeg_data"
MODEL_PATH = "eeg_thought_model.h5"
WINDOW_SIZE = 128
STRIDE = 64
COLS = ['TP9', 'AF7', 'AF8', 'TP10']
DURATION = 30  # 수집 시간 (초)


# --------------------------------------------


# --------------------------------------------
# 1. muselsl stream 실행
def start_stream():
    print("🧠 muselsl stream 시작 중...")
    process = subprocess.Popen(["muselsl", "stream"])
    time.sleep(5)
    return process


def stop_stream(process):
    print("🛑 muselsl stream 종료 중...")
    process.terminate()


# --------------------------------------------


# --------------------------------------------
# 2. EEG 데이터 수집
def collect_data(label):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    filename = os.path.join(DATA_FOLDER, f"{label}_{int(time.time())}.csv")

    print("🔍 EEG 스트림 탐색 중...")
    streams = resolve_byprop('type', 'EEG', timeout=10)
    if not streams:
        raise RuntimeError("❌ EEG 스트림을 찾을 수 없습니다.")

    inlet = StreamInlet(streams[0])
    print(f"✅ EEG 스트림 연결됨. {label} 라벨 데이터 {DURATION}초 동안 수집 중...")

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(COLS + ['timestamp', 'label'])
        start_time = time.time()
        while time.time() - start_time < DURATION:
            sample, timestamp = inlet.pull_sample()
            writer.writerow(sample + [timestamp, label])

    print(f"✅ 저장 완료: {filename}")


# --------------------------------------------


# --------------------------------------------
# 3. 데이터 전처리 (윈도우로 쪼개기)
def load_and_preprocess_data():
    X, y = [], []

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_FOLDER, file))
            if not set(COLS).issubset(df.columns): continue

            signals = df[COLS].values
            label = df['label'].values[0]

            for i in range(0, len(signals) - WINDOW_SIZE, STRIDE):
                window = signals[i:i + WINDOW_SIZE]
                X.append(window)
                y.append(label)

    X = np.array(X)
    y = LabelEncoder().fit_transform(y)
    print(f"✅ 총 샘플 수: {len(X)}, 클래스: {set(y)}")
    return X, y


# --------------------------------------------


# --------------------------------------------
# 4. 모델 정의 + 학습
def build_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv1D(64, 5, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = build_model(X.shape[1:], num_classes=len(set(y)))
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
    model.save(MODEL_PATH)
    print(f"✅ 모델 저장 완료: {MODEL_PATH}")


# --------------------------------------------


# --------------------------------------------
# 5. 실시간 추론
def predict_thought():
    print("🔍 EEG 스트림 연결 중 (실시간 추론)...")
    streams = resolve_byprop('type', 'EEG', timeout=10)
    inlet = StreamInlet(streams[0])

    model = tf.keras.models.load_model(MODEL_PATH)
    buffer = deque(maxlen=WINDOW_SIZE)

    print("🧠 실시간 추론 시작! (생각해보세요...)")

    while True:
        sample, _ = inlet.pull_sample()
        buffer.append(sample)

        if len(buffer) == WINDOW_SIZE:
            X_input = np.expand_dims(np.array(buffer), axis=0)
            pred = model.predict(X_input, verbose=0)
            pred_label = np.argmax(pred)
            confidence = np.max(pred)
            print(f"🧠 예측: {pred_label} | 확률: {confidence:.2f}")
            time.sleep(0.5)


# --------------------------------------------


# --------------------------------------------
# 📌 실행 흐름
if __name__ == "__main__":
    try:
        stream_proc = start_stream()

        # 1. 데이터 수집
        while True:
            label = input("수집할 라벨 입력 (예: a, rest), 또는 엔터 시 종료: ").strip()
            if not label:
                break
            collect_data(label)

        stop_stream(stream_proc)

        # 2. 전처리 + 학습
        print("\n📊 데이터 로딩 및 전처리 중...")
        X, y = load_and_preprocess_data()
        train_model(X, y)

        # 3. 실시간 추론 시작
        input("\n▶️ [Enter] 키를 누르면 실시간 추론을 시작합니다...")
        stream_proc = start_stream()
        predict_thought()

    except KeyboardInterrupt:
        stop_stream(stream_proc)
        print("\n🛑 사용자에 의해 중단됨.")
    except Exception as e:
        stop_stream(stream_proc)
        print(f"❌ 오류 발생: {e}")