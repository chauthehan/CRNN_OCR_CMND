# CRNN_OCR_CMND

CRNN là một mạng kết hợp giữa CNN và RNN để xử lý những ảnh chứa các thông tin dạng chuỗi như là chữ viết. Nó chủ yếu được sử dụng cho công nghệ OCR và có các điểm mạnh sau:
 1. End-to-end learning
 2. Độ dài của nhãn là ngẫu nhiên 
 3. Không cần các kỹ thuật phát hiện và cắt từng ký tự một
 
 Bạn có thể sử dụng CRNN dể OCR, đọc biển số xe, nhận dạng text,.. Phụ thuộc vào dữ liệu của bạn. Tôi sử dụng một phiên bản được chỉnh sửa nhẹ so với bản gốc trong paper https://arxiv.org/abs/1507.05717  khi thêm vào 3 lớp convolutional và đầu vào sẽ là 300 *32

Sử dụng:

File dic.txt chứa danh sách các chữ cái tiếng việt, tạo ra từ create_dic.py
Command: "python create_dic.py"

Dữ liệu được chia thành 3 file train test và val với format mẫu trong git. Sử dụng buid_dataset.py để lưu data dưới dạng file hdf5. Khi build file hdf5, ảnh sẽ được để ngược 90 độ cho phù hợp với timesteps.
Command: "python build_dataset.py --dataset "PATH_TO_DATASET"
Note: Đổi tên file hdf5 trong file python tùy mục đích sử dụng

File count_char.py để đếm độ dài lớn nhất của nhãn.
Note: Sửa đường dẫn trong file để file chứa data để sử dụng.

Tạo dữ liệu giả là các text box với nền của CMND sử dụng trdg https://github.com/Belval/TextRecognitionDataGenerator    (tự tạo nền của cho ảnh)

Quá trình train: sử dụng ELU() tốt hơn so với RELU và ADAM thay vì SGD để loss xuống thấp nhất
                 
Kết quả: độ chính xác trên tập test( tập test sử dụng ảnh thật và gán nhãn bằng tay) đạt 70%. 
