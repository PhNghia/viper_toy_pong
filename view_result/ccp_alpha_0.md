(viper-env) C:\SGU-LEARN\nam4_hk2\seminar\viper-verifiable-rl-impl>python main.py train-viper --verbose 2 --env-name ToyPong-v0 --n-env 4 --ccp-alpha 0 --total-timesteps 1_000_000
Training Viper on ToyPong-v0
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Policy score: 826.5600 +/- 320.3648
Viper iteration complete. Dataset size: 1000000
Best policy:    0
Mean reward:    826.5600
Max depth:      31
# Leaves:       1546
Saving to       ./log/viper_ToyPong-v0_0.0_all-leaves_all-depth.joblib

-----------------------------------------------------------------------------------------------
Khi `ccp_alpha = 0`, cây quyết định (Decision Tree) sẽ không thực hiện bất kỳ việc **cắt tỉa** nào, tức là cây sẽ phát triển đến mức tối đa với tất cả các nhánh mà không bị giới hạn. Điều này có thể dẫn đến một số vấn đề như:

### **1. Cây quyết định quá phức tạp (Overfitting)**
- Khi `ccp_alpha = 0`, cây quyết định sẽ không bị **cắt tỉa** để loại bỏ các nhánh ít quan trọng. Điều này có thể khiến cây học rất chi tiết các mẫu trong dữ liệu huấn luyện, dẫn đến **overfitting** — nghĩa là cây học rất sâu và phức tạp, nhưng chỉ có thể hoạt động tốt với dữ liệu huấn luyện, không thể tổng quát tốt với các tình huống mới.
- Trong trường hợp này, cây quyết định học được rất nhiều chi tiết nhỏ và giống hệt nhau, khiến cho các chính sách (policy) từ cây này cũng sẽ giống nhau qua các iteration. Điều này dẫn đến việc **dataset thu thập được** cho các iteration sau sẽ không thay đổi nhiều, do đó các cây quyết định (policies) sau đó không cải thiện được nhiều, tạo ra kết quả **giống nhau**.

### **2. Dễ gây ra **overfitting** trên các chính sách**
- Khi không có việc cắt tỉa, cây quyết định có thể học những quy tắc **quá phức tạp**, dẫn đến việc các quyết định mà cây đưa ra (tức là **actions**) sẽ ít thay đổi hoặc gần như giống nhau. Điều này khiến cho **các giá trị trong dataset thu thập được** gần như giống nhau qua nhiều iteration, dẫn đến việc các chính sách mà bạn huấn luyện sẽ không được cải thiện nhiều, tạo ra kết quả giống nhau.

### **3. Không có sự thay đổi trong cấu trúc cây**
- Với `ccp_alpha = 0`, cây sẽ không bị "cắt" ở các điểm không quan trọng, dẫn đến việc cây học quá sâu và chi tiết. Khi bạn cập nhật cây quyết định qua các iteration, nếu không có cắt tỉa, các cây này sẽ phát triển theo cùng một cách và kết quả cuối cùng (policy score) có thể rất giống nhau.

### **Giải thích ngắn gọn về `ccp_alpha`:**
- **`ccp_alpha = 0`**: Cây quyết định không bị cắt tỉa, dễ dẫn đến việc cây quá phức tạp và không thể học được nhiều sự khác biệt, tạo ra các chính sách giống nhau.
- **`ccp_alpha > 0`**: Cây sẽ thực hiện cắt tỉa, loại bỏ các nhánh không quan trọng và giúp cây học được những mô hình đơn giản hơn, giảm overfitting và giúp các chính sách có thể khác nhau hơn qua các iteration.

### **Vì sao khi `ccp_alpha = 0.0001`, kết quả lại khác nhau?**
Khi bạn đặt `ccp_alpha = 0.0001`, việc **cắt tỉa cây quyết định** sẽ giúp giảm đi mức độ phức tạp của cây, khiến nó học được những đặc điểm chung của dữ liệu mà không phải học quá chi tiết các yếu tố không quan trọng. Điều này giúp các chính sách (policies) có sự thay đổi và cải thiện qua các iteration, tạo ra sự đa dạng và không bị giống nhau hoàn toàn.

---

### **Tóm lại:**
- **`ccp_alpha = 0`** khiến cây quyết định không bị cắt tỉa và dễ dẫn đến overfitting, kết quả là các chính sách học được giống nhau qua các iteration.
- **`ccp_alpha = 0.0001`** tạo ra việc cắt tỉa cây quyết định, giúp cây học những mô hình đơn giản và hiệu quả hơn, từ đó cải thiện kết quả và tạo sự thay đổi trong các chính sách.

------------------------------------------
(viper-env) C:\SGU-LEARN\nam4_hk2\seminar\viper-verifiable-rl-impl>python main.py train-viper --verbose 2 --env-name ToyPong-v0 --n-env 4 --ccp-alpha 0 --total-timesteps 1_000_000
Training Viper on ToyPong-v0
Policy score: 759.4000 +/- 362.3062
Policy score: 593.4200 +/- 401.4065
Policy score: 833.9500 +/- 341.5114
Policy score: 947.5300 +/- 184.5486
Policy score: 870.2800 +/- 309.4674
Policy score: 951.4400 +/- 198.7953
Policy score: 867.0200 +/- 280.4041
Policy score: 947.3500 +/- 188.1066
Policy score: 983.1500 +/- 119.1747
Policy score: 960.4100 +/- 164.3750
Policy score: 961.7900 +/- 168.7549
Policy score: 977.2800 +/- 127.4050
Policy score: 961.4400 +/- 170.1072
Policy score: 964.7000 +/- 173.8389
Policy score: 928.4800 +/- 239.1004
Policy score: 993.1600 +/- 53.3880
Policy score: 972.7100 +/- 149.8732
Policy score: 990.3800 +/- 95.7178
Policy score: 985.8700 +/- 100.1590
Policy score: 991.4300 +/- 85.2704
Policy score: 990.2100 +/- 97.4093
Policy score: 972.8900 +/- 154.8608
Policy score: 988.5100 +/- 88.6433
Policy score: 982.5400 +/- 108.6269
Policy score: 981.8300 +/- 127.3319
Policy score: 985.1400 +/- 94.0296
Policy score: 988.2600 +/- 84.5646
Policy score: 984.1000 +/- 111.3058
Policy score: 964.4800 +/- 175.0941
Policy score: 1000.0000 +/- 0.0000
Policy score: 981.4000 +/- 109.1395
Policy score: 1000.0000 +/- 0.0000
Policy score: 977.8000 +/- 126.4703
Policy score: 991.3000 +/- 61.2772
Policy score: 992.1000 +/- 78.6040
Policy score: 990.2200 +/- 97.3098
Policy score: 983.7200 +/- 106.1087
Policy score: 1000.0000 +/- 0.0000
Policy score: 996.4400 +/- 35.4216
Policy score: 1000.0000 +/- 0.0000
Policy score: 972.2100 +/- 139.6494
Policy score: 981.8600 +/- 127.0055
Policy score: 992.7300 +/- 72.3356
Policy score: 988.6400 +/- 89.7338
Policy score: 1000.0000 +/- 0.0000
Policy score: 994.4300 +/- 55.4208
Policy score: 971.2800 +/- 163.3829
Policy score: 980.7400 +/- 134.8664
Policy score: 963.5300 +/- 179.1913
Policy score: 995.0000 +/- 49.7494
Policy score: 1000.0000 +/- 0.0000
Policy score: 961.3600 +/- 170.0039
Policy score: 985.9700 +/- 103.8865
Policy score: 993.6300 +/- 63.3807
Policy score: 980.4000 +/- 137.2073
Policy score: 1000.0000 +/- 0.0000
Policy score: 996.3300 +/- 36.5160
Policy score: 992.0600 +/- 79.0020
Policy score: 990.0700 +/- 98.8023
Policy score: 1000.0000 +/- 0.0000
Policy score: 990.8500 +/- 91.0414
Policy score: 1000.0000 +/- 0.0000
Policy score: 1000.0000 +/- 0.0000
Policy score: 993.9500 +/- 60.1967
Policy score: 999.4600 +/- 5.3729
Policy score: 995.3300 +/- 46.4659
Policy score: 1000.0000 +/- 0.0000
Policy score: 983.1300 +/- 119.6121
Policy score: 1000.0000 +/- 0.0000
Policy score: 992.3700 +/- 75.9175
Policy score: 990.7900 +/- 91.6383
Policy score: 985.8100 +/- 103.2592
Policy score: 995.4900 +/- 44.8739
Policy score: 991.3600 +/- 72.1354
Policy score: 998.4600 +/- 15.3228
Policy score: 1000.0000 +/- 0.0000
Policy score: 990.0800 +/- 98.7028
Policy score: 994.2600 +/- 57.1123
Policy score: 1000.0000 +/- 0.0000
Policy score: 1000.0000 +/- 0.0000
Viper iteration complete. Dataset size: 1000000
Best policy:    29
Mean reward:    1000.0000
Max depth:      37
# Leaves:       13901
Saving to       ./log/viper_ToyPong-v0_0.0_all-leaves_all-depth.joblib