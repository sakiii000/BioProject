from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import numpy as np
import esm
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://sakiii000.github.io"]) 
# 載入模型和相關組件
device = torch.device('cpu')

# CNN 模型定義
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
try:
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.eval()
    model = model.to(device)
except Exception as e:
    print(f"載入 ESM 模型時發生錯誤：{str(e)}")
    exit(1)

# 載入訓練好的 CNN 模型
input_shape = (1, 480)  # ESM-2 特徵維度
num_classes = 2  # SNARE/非SNARE
cnn_model = CNN(input_shape, num_classes).to(device)
# cnn_model.load_state_dict(torch.load('best_cnn_model.pth'))
cnn_model.load_state_dict(torch.load('best_cnn_model.pth', map_location='cpu'))


cnn_model.eval()

def get_esm_embedding(sequence):
    """使用 ESM-2 模型獲取序列的嵌入向量"""
    # 將序列轉換為模型可接受的格式
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
    batch_tokens = batch_tokens.to(device)
    
    # 獲取嵌入向量
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])
    
    # 獲取最後一層的嵌入向量
    embeddings = results["representations"][12]
    
    # 移除特殊標記（CLS和EOS）的嵌入
    embeddings = embeddings[0, 1:-1, :]
    
    # 計算平均嵌入向量
    mean_embedding = embeddings.mean(dim=0)
    
    return mean_embedding.cpu().numpy()

def predict_sequence(sequence):
    """預測序列是否為 SNARE"""
    # 獲取 ESM 嵌入
    embedding = get_esm_embedding(sequence)
    
    # 重塑為模型輸入格式
    embedding = embedding.reshape(1, 1, -1)
    embedding = torch.FloatTensor(embedding).to(device)
    
    # 進行預測
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
    """發送預測結果到指定郵箱"""
    # 設置郵件內容
    msg = MIMEMultipart()
    msg['From'] = 'brian20040211@gmail.com'  # 替換為您的 Gmail
    msg['To'] = to_email
    msg['Subject'] = 'SNARE Protein Prediction Results'
    
    # 郵件正文
    body = f"""
    Protein Sequence: {sequence}
    
    Prediction Results:
    - Prediction: {result['prediction']}
    - Confidence: {result['confidence']*100:.1f}%
    - SNARE Probability: {result['probabilities']['SNARE']*100:.1f}%
    - Non-SNARE Probability: {result['probabilities']['Non-SNARE']*100:.1f}%
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    # 發送郵件
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('brian20040211@gmail.com', 'tpfg iwuy fybt dnjj')  # 替換為您的 Gmail 和應用密碼
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip()
        email = data.get('email', '').strip()
        
        if not sequence:
            return jsonify({'error': 'No sequence provided'}), 400
        
        if not email:
            return jsonify({'error': 'No email provided'}), 400
        
        # 清理序列（只保留氨基酸字符）
        sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
        
        if not sequence:
            return jsonify({'error': 'Invalid sequence'}), 400
        
        # 進行預測
        result = predict_sequence(sequence)
        
        # 發送郵件
        if send_email(email, sequence, result):
            result['email_sent'] = True
        else:
            result['email_sent'] = False
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
