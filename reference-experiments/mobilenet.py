"""
MobileNet V2 Training Script for Google Colab
Binary Classification: empty vs not_empty
Google Drive 경로: /content/drive/MyDrive/2ㄷ1/
자동 Train:Test = 8:2 분할
** Val Loss 최소 지점 모델 저장 **
"""

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

# 설정
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# 데이터 경로 설정
BASE_DIR = '/content/drive/MyDrive/2ㄷ1'
EMPTY_DIR = os.path.join(BASE_DIR, 'empty')
NOT_EMPTY_DIR = os.path.join(BASE_DIR, 'not_empty')

# 임시 train/test 폴더 경로
TRAIN_DIR = '/content/train_split'
TEST_DIR = '/content/test_split'

print("=== 경로 확인 ===")
print(f"Base 폴더 존재: {os.path.exists(BASE_DIR)}")
print(f"empty 폴더 존재: {os.path.exists(EMPTY_DIR)}")
print(f"not_empty 폴더 존재: {os.path.exists(NOT_EMPTY_DIR)}")

# ============================================================
# 데이터 분할: empty와 not_empty 각각 8:2로 나누기
# ============================================================
def split_data(source_dir, train_dest, test_dest, test_size=0.2, seed=42):
    """
    폴더 내 이미지를 train/test로 분할

    Args:
        source_dir: 원본 폴더 (empty 또는 not_empty)
        train_dest: train 목적지 폴더
        test_dest: test 목적지 폴더
        test_size: test 비율 (0.2 = 20%)
        seed: 랜덤 시드
    """
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)

    files = [f for f in os.listdir(source_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]

    if len(files) == 0:
        print(f"⚠️  {source_dir}에 이미지가 없습니다!")
        return 0, 0

    train_files, test_files = train_test_split(
        files,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    for fname in train_files:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(train_dest, fname)
        shutil.copy2(src, dst)

    for fname in test_files:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(test_dest, fname)
        shutil.copy2(src, dst)

    return len(train_files), len(test_files)

print("\n=== 데이터 분할 (Train:Test = 8:2) ===")

if os.path.exists(TRAIN_DIR):
    shutil.rmtree(TRAIN_DIR)
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)

print("Empty 데이터 분할 중...")
train_empty, test_empty = split_data(
    EMPTY_DIR,
    os.path.join(TRAIN_DIR, 'empty'),
    os.path.join(TEST_DIR, 'empty'),
    test_size=0.2
)
print(f"  Train: {train_empty}개, Test: {test_empty}개")

print("Not_empty 데이터 분할 중...")
train_not_empty, test_not_empty = split_data(
    NOT_EMPTY_DIR,
    os.path.join(TRAIN_DIR, 'not_empty'),
    os.path.join(TEST_DIR, 'not_empty'),
    test_size=0.2
)
print(f"  Train: {train_not_empty}개, Test: {test_not_empty}개")

print(f"\n전체 요약:")
print(f"  Train 전체: {train_empty + train_not_empty}개")
print(f"  Test 전체: {test_empty + test_not_empty}개")
print(f"  비율: {(train_empty + train_not_empty) / (train_empty + train_not_empty + test_empty + test_not_empty) * 100:.1f}% : {(test_empty + test_not_empty) / (train_empty + train_not_empty + test_empty + test_not_empty) * 100:.1f}%")

print("\nTensorFlow 버전:", tf.__version__)
print("GPU 사용 가능 여부:", tf.config.list_physical_devices('GPU'))

# ============================================================
# 데이터 증강 설정
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("\n=== 데이터셋 정보 ===")
print(f"훈련 샘플 수: {train_generator.samples}")
print(f"검증 샘플 수: {validation_generator.samples}")
print(f"테스트 샘플 수: {test_generator.samples}")
print(f"클래스: {train_generator.class_indices}")

# ============================================================
# MobileNetV2 모델 구축 - Dense Layer 파라미터 대폭 증가
# ============================================================
def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35
    )

    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        
        # 첫 번째 Dense Block (1024 뉴런)
        layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # 두 번째 Dense Block (512 뉴런)
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 세 번째 Dense Block (256 뉴런)
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # 출력층
        layers.Dense(1, activation='sigmoid')
    ])

    return model, base_model

model, base_model = build_model()

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

print("\n=== 모델 구조 ===")
model.summary()

trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\n📊 Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

# 결과 저장 경로
SAVE_DIR = '/content/drive/MyDrive/2ㄷ1'

# ============================================================
# 콜백 설정 - Val Loss 최소 지점 저장
# ============================================================
callbacks = [
    # ⭐ Validation Loss가 가장 낮은 모델 저장 (메인)
    keras.callbacks.ModelCheckpoint(
        f'{SAVE_DIR}/best_model_val_loss.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1,
        save_weights_only=False
    ),
    # Validation Accuracy가 가장 높은 모델도 별도 저장 (참고용)
    keras.callbacks.ModelCheckpoint(
        f'{SAVE_DIR}/best_model_val_acc.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=False
    ),
    # 조기 종료 - val_loss 기준, 최적 가중치 복원
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,  # 7 epoch 동안 개선 없으면 중지
        restore_best_weights=True,  # ⭐ 최적 가중치 자동 복원
        verbose=1,
        mode='min'
    ),
    # 학습률 감소
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================
# 1단계: 전이 학습 (베이스 모델 동결)
# ============================================================
print("\n" + "="*60)
print("=== 1단계: 전이 학습 시작 (베이스 모델 동결) ===")
print("="*60)
history1 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 2단계: 미세 조정 (베이스 모델의 일부 레이어 해동)
# ============================================================
print("\n" + "="*60)
print("=== 2단계: 미세 조정 시작 (베이스 모델 일부 해동) ===")
print("="*60)
base_model.trainable = True

fine_tune_at = len(base_model.layers) - 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

trainable_params_ft = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\n📊 Fine-tuning Trainable Parameters: {trainable_params_ft:,} ({trainable_params_ft/1e6:.2f}M)")

history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 학습 결과 시각화
# ============================================================
def plot_training_history(history1, history2):
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', label='Fine-tuning starts')
    
    # Val Loss 최소 지점 표시
    min_val_loss_idx = np.argmin(val_loss)
    plt.axvline(x=min_val_loss_idx, color='g', linestyle=':', linewidth=2, label=f'Best Val Loss (epoch {min_val_loss_idx+1})')
    
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=len(history1.history['loss']), color='r', linestyle='--', label='Fine-tuning starts')
    
    # Val Loss 최소 지점 표시
    plt.axvline(x=min_val_loss_idx, color='g', linestyle=':', linewidth=2, label=f'Best Val Loss (epoch {min_val_loss_idx+1})')
    plt.scatter([min_val_loss_idx], [val_loss[min_val_loss_idx]], color='g', s=100, zorder=5, label=f'Min Val Loss: {val_loss[min_val_loss_idx]:.4f}')
    
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 3, 3)
    if 'lr' in history1.history:
        lr = history1.history['lr'] + history2.history['lr']
        plt.plot(epochs_range, lr)
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n⭐ Val Loss 최소 지점: Epoch {min_val_loss_idx+1}, Loss = {val_loss[min_val_loss_idx]:.4f}")

plot_training_history(history1, history2)

# ============================================================
# 최적 모델 로드 및 평가
# ============================================================
print("\n" + "="*60)
print("=== 최적 모델 (Val Loss 최소) 로드 및 평가 ===")
print("="*60)

# Val Loss 최소 모델 로드
best_model = keras.models.load_model(f'{SAVE_DIR}/best_model_val_loss.h5')

test_loss, test_acc, test_precision, test_recall = best_model.evaluate(test_generator)
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)

print(f"\n최종 테스트 결과 (Val Loss 최소 모델):")
print(f"정확도 (Accuracy): {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"정밀도 (Precision): {test_precision:.4f}")
print(f"재현율 (Recall): {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")

# ============================================================
# Confusion Matrix 생성
# ============================================================
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

test_generator.reset()
predictions = best_model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = test_generator.classes

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['empty', 'not_empty'],
            yticklabels=['empty', 'not_empty'])
plt.title('Confusion Matrix (Best Val Loss Model)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(f'{SAVE_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 분류 리포트 ===")
print(classification_report(true_classes, predicted_classes,
                          target_names=['empty', 'not_empty']))

# ============================================================
# 최종 모델 저장
# ============================================================
best_model.save(f'{SAVE_DIR}/mobilenet_v2_final.h5')
print(f"\n최적 모델이 '{SAVE_DIR}/mobilenet_v2_final.h5'에 저장되었습니다.")

# ============================================================
# 예측 예시 함수
# ============================================================
def predict_image(image_path):
    """단일 이미지 예측 함수"""
    img = keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = best_model.predict(img_array)[0][0]
    class_name = 'not_empty' if prediction > 0.5 else 'empty'
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return class_name, confidence

print("\n=== 샘플 예측 테스트 ===")
try:
    import glob
    sample_images = glob.glob(f'{TEST_DIR}/empty/*.jpg')[:3] + glob.glob(f'{TEST_DIR}/not_empty/*.jpg')[:3]

    if sample_images:
        for img_path in sample_images:
            class_pred, conf = predict_image(img_path)
            true_label = 'empty' if '/empty/' in img_path else 'not_empty'
            correct = '✅' if class_pred == true_label else '❌'
            print(f"{correct} {os.path.basename(img_path)}")
            print(f"   실제: {true_label} | 예측: {class_pred} (신뢰도: {conf:.2%})")
except Exception as e:
    print(f"샘플 예측 테스트 중 오류: {e}")

print("\n" + "="*60)
print("✅ 훈련 완료!")
print("="*60)
print(f"📁 저장된 파일:")
print(f"   - best_model_val_loss.h5: ⭐ Val Loss 최소 모델 (메인)")
print(f"   - best_model_val_acc.h5: Val Accuracy 최대 모델 (참고)")
print(f"   - mobilenet_v2_final.h5: 최종 복사본")
print(f"   - training_history.png: 학습 곡선")
print(f"   - confusion_matrix.png: 혼동 행렬")
print("="*60)

print("\n임시 폴더 정리 중...")
try:
    shutil.rmtree(TRAIN_DIR)
    shutil.rmtree(TEST_DIR)
    print("임시 폴더 삭제 완료")
except:
    print("임시 폴더 삭제 실패 (무시 가능)")