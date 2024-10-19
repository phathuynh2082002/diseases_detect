from flask import Flask, request, jsonify
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Tạo ứng dụng Flask
app = Flask(__name__)
CORS(app)
# Tải mô hình đã huấn luyện
model = load_model('model.h5')
# Danh sách các nhãn lớp
folders = [
    'doi_duc_qua', 'la_khoe', 'ruoi_vang', 'sau_can_la',
    'thoi_nhun', 'trai_khoe', 'trai_non_khoe', 'vop_la'
]


def decode_image(base64_str):
    # Giải mã chuỗi Base64
    decoded_bytes = base64.b64decode(base64_str)
    img = Image.open(BytesIO(decoded_bytes))
    img = img.resize((300, 300))  # Chỉnh kích thước ảnh về 300x300
    img_array = img_to_array(img)  # Chuyển đổi ảnh thành numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
    return img_array

@app.route('/', methods=['GET'])
def test():
    return '123'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image part in the request"}), 400

    # Giải mã và tiền xử lý ảnh
    base64_image = data['image']
    img_array = decode_image(base64_image)

    # Dự đoán lớp của ảnh
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_probabilities = prediction[0]  # Lấy xác suất của tất cả các lớp
    predicted_probability = predicted_probabilities[predicted_class[0]]  # Xác suất cho lớp dự đoán

    # Kiểm tra độ chính xác
    if predicted_probability < 0.8:
        return jsonify("unpredictable")

    # Lấy tên thư mục tương ứng với chỉ số lớp được dự đoán
    predicted_folder_name = folders[predicted_class[0]]

    # Trả về kết quả dự đoán dưới dạng JSON
    return jsonify(predicted_folder_name)


if __name__ == '__main__':
    app.run(debug=True)