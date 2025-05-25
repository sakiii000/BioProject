import streamlit as st
import torch
import torch.nn as nn
import esm
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re

# 裝置設定
device = torch.device('cpu')

# CNN 模型
class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[0], 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.flat_features = input_shape[1] * 256 // 8
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
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
@st.cache_resource
def load_esm_model():
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.eval().to(device)
    return model, alphabet

# 載入訓練好的 CNN 模型
@st.cache_resource
def load_cnn_model():
    model = CNN((1, 480), 2).to(device)
    model.load_state_dict(torch.load("best_cnn_model.pth", map_location=device))
    model.eval()
    return model

# 嵌入序列
def get_esm_embedding(sequence, esm_model, alphabet):
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("protein", sequence)])
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[12])
    embeddings = results["representations"][12]
    embeddings = embeddings[0, 1:-1, :]  # 去除 CLS 和 EOS
    mean_embedding = embeddings.mean(dim=0)
    return mean_embedding.cpu().numpy()

# 預測功能
def predict_sequence(sequence, esm_model, alphabet, cnn_model):
    embedding = get_esm_embedding(sequence, esm_model, alphabet)
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

# 寄送 Email
def send_email(to_email, sequence, result):
    msg = MIMEMultipart()
    msg['From'] = 'brian20040211@gmail.com'
    msg['To'] = to_email
    msg['Subject'] = 'SNARE Protein Prediction Results'

    body = f"""\
Protein Sequence: {sequence}

Prediction Results:
- Prediction: {result['prediction']}
- Confidence: {result['confidence'] * 100:.1f}%
- SNARE Probability: {result['probabilities']['SNARE'] * 100:.1f}%
- Non-SNARE Probability: {result['probabilities']['Non-SNARE'] * 100:.1f}%
"""

    # 指定編碼為 utf-8
    text_part = MIMEText(body, 'plain', 'utf-8')
    msg.attach(text_part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('brian20040211@gmail.com', 'tpfg iwuy fybt dnjj')  # 請用應用程式密碼
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email sending error: {str(e)}")
        return False


# ========== Streamlit UI ==========
st.markdown("""
    <style>
    .stApp {
        background-color: #c5cae9;
    }
    .main {
        background-color: #e8eaf6;
        padding: 40px;  /* 原本是 300px，改成更合理的值 */
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 40px auto;
        max-width: 700px;
    }
    h2 {
      margin-bottom: 20px;
      color: #3f51b5;
    }
    label, .stTextInput > label, .stTextArea > label {
        font-weight: bold;
        color: #303f9f;
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #e8eaf6;
        border-radius: 6px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    .stButton button {
        background: linear-gradient(135deg, #5c6bc0, #3f51b5);
        color: white;
        border-radius: 25px;
        font-size: 16px;
        padding: 10px;
        border: none;
        width: 100%;
        transition: background 0.3s, transform 0.2s;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #7986cb, #5c6bc0);
    }
    .stButton button:active {
        transform: scale(0.97);
    }
    </style>
""", unsafe_allow_html=True)



# ========== Streamlit 表單 UI ========== #
with st.container():
    #st.markdown("""
    #    <div class="main">
    #        <h2>Predict SNARE Proteins</h2>
    #        <p>請輸入蛋白質序列，我們將預測是否為 SNARE 並寄送至您的信箱。</p>
    #""", unsafe_allow_html=True)
    st.markdown("## Predict SNARE Proteins")
    st.markdown("請輸入蛋白質序列，我們將預測是否為 SNARE 並寄送至您的信箱。")
    email = st.text_input("📧 請輸入您的 Email")
    sequence = st.text_area("🔢 請輸入蛋白質序列（A-Z 氨基酸字母）", height=150)

    if st.button("Submit"):
        sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())

        if not sequence or not email:
            st.warning("請輸入有效的序列與電子郵件")
        else:
            with st.spinner("模型運算中，請稍候..."):
                esm_model, alphabet = load_esm_model()
                cnn_model = load_cnn_model()
                result = predict_sequence(sequence, esm_model, alphabet, cnn_model)

            st.success("✅ 預測完成！")
            st.write(f"**預測結果**: {result['prediction']}")
            st.write(f"**信心指數**: {result['confidence']*100:.1f}%")
            st.write(f"**SNARE 機率**：{result['probabilities']['SNARE'] * 100:.1f}%")
            st.write(f"**Non-SNARE 機率**：{result['probabilities']['Non-SNARE'] * 100:.1f}%")

            if send_email(email, sequence, result):
                st.success("📬 預測結果已成功寄出！")
            else:
                st.warning("❗ 郵件寄送失敗，請確認信箱或稍後再試。")

    # 結尾的 </div> 放這裡
    #st.markdown("</div>", unsafe_allow_html=True)
