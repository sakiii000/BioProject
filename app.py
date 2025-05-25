import streamlit as st
import torch
import torch.nn as nn
import esm
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np

# 初始化模型
device = torch.device('cpu')

class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[0], 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.flat_features = input_shape[1] * 256 // 8
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 載入 ESM 模型
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
model = model.eval().to(device)

# 載入 CNN 模型
input_shape = (1, 480)
cnn_model = CNN(input_shape, 2).to(device)
cnn_model.load_state_dict(torch.load('best_cnn_model.pth', map_location='cpu'))
cnn_model.eval()

def get_esm_embedding(sequence):
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("protein", sequence)])
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])
    embeddings = results["representations"][12]
    embeddings = embeddings[0, 1:-1, :]
    return embeddings.mean(dim=0).cpu().numpy()

def predict_sequence(sequence):
    embedding = get_esm_embedding(sequence)
    embedding = embedding.reshape(1, 1, -1)
    embedding = torch.FloatTensor(embedding).to(device)
    with torch.no_grad():
        outputs = cnn_model(embedding)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)
        confidence = probabilities[0][prediction].item()
    return {
        'prediction': 'SNARE' if prediction.item() == 1 else 'Non-SNARE',
        'confidence': confidence,
        'probabilities': {
            'SNARE': probabilities[0][1].item(),
            'Non-SNARE': probabilities[0][0].item()
        }
    }

def send_email(to_email, sequence, result):
    msg = MIMEMultipart()
    msg['From'] = 'brian20040211@gmail.com'
    msg['To'] = to_email
    msg['Subject'] = 'SNARE Protein Prediction Results'
    body = f"""
    Protein Sequence: {sequence}

    Prediction Results:
    - Prediction: {result['prediction']}
    - Confidence: {result['confidence']*100:.1f}%
    - SNARE Probability: {result['probabilities']['SNARE']*100:.1f}%
    - Non-SNARE Probability: {result['probabilities']['Non-SNARE']*100:.1f}%
    """
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('brian20040211@gmail.com', '你的應用密碼')  # 務必替換！
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

# Streamlit 介面
st.title("SNARE Protein Predictor")
email = st.text_input("輸入你的 Email")
sequence = st.text_area("輸入蛋白質序列（僅限 20 種氨基酸）", height=200)

if st.button("開始預測"):
    if not sequence or not email:
        st.error("請輸入蛋白質序列與 Email")
    else:
        cleaned_seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
        if not cleaned_seq:
            st.error("序列無效，請重新輸入")
        else:
            st.info("正在進行預測，請稍候...")
            result = predict_sequence(cleaned_seq)
            st.success(f"預測結果：{result['prediction']}")
            st.write(f"信心分數：{result['confidence']*100:.1f}%")
            st.write(f"SNARE 機率：{result['probabilities']['SNARE']*100:.1f}%")
            st.write(f"Non-SNARE 機率：{result['probabilities']['Non-SNARE']*100:.1f}%")

            if send_email(email, cleaned_seq, result):
                st.success("結果已發送至您的 Email")
            else:
                st.warning("Email 發送失敗，請稍後再試")
