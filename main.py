from pydub import AudioSegment
import numpy as np
import soundfile as sfile
import time

#xu ly am thanh
#ham chuyen doi sang don vi dB
def decibel(arr):
            ref = 1
            if arr != 0:
                return 20 * np.log10(np.abs(arr) / ref)
            else:
                return -60
def data_process(filename):
    x_input = []
    for i in filename:
        data_train = []
        signal, sr = sfile.read(i)
        try:
            samples_sf = signal[:, 0]  
        except:
            samples_sf = signal

        data=[decibel(j) for j in samples_sf]
        data_train.append(round(np.mean(data), 2)) #trung binh
        data_train.append(round(np.median(data), 2))#trung vi
        data_train.append(round(np.std(data), 2))#Do lech chuan
        data_train.append(round(np.var(data), 2))# Phuong sai
        x_input.append(data_train)
    return x_input

#mo hinh no-ron
def hard_lim(x):
    if x >= 0:
        return 1
    else:
        return 0
class Neural:
    def __init__(self, x_inputs, w_weight, bias):
        self.x_inputs = x_inputs
        self.w_weight = w_weight
        self.bias = bias
    def Sum_function(self):
        n = self.bias
        for x, w in zip(self.x_inputs, self.w_weight):
            n += x*w
        return n
    def output(self):
        a = hard_lim(Neural.Sum_function(self))
        return a
def e(a, target):
    e = target - a
    return e
def update_weight(w_old, X, e):
    w_new = []
    for w, x in zip(w_old, X):
        i = w +(e * x)
        w_new.append(i)
    return w_new
def update_bias(b_old, e):
    b_new = b_old + e
    return b_new


def TEST(file, w_weight, bias):
    print('OK ĐỂ LẤY DỮ LIỆU........')
    file_n = []
    try:
        file_n.append(file)
        ex = data_process(file_n)
        neural = Neural(ex[0], w_weight, bias)
        a = neural.output()

        if (a == 0) :
            print(file, '==> TIẾNG CON VỊT CHỨ CÒN GÌ NỮA!')
        if (a == 1):
            print(file, '==> TIẾNG CON GÀ CHẮN CHẮN LUÔN!!!')
    except:
        print("da xay ra loi.")

    print('Đã hoàn thành xuất sắc nhiệm vụ HOAN HÔ!')
    print("===============***================")
    print()
    print('Bạn có muốn thực hiện tiếp không?')
    print('Chọn 1 để tiến hành nhận dạng, chọn 0 nếu muốn rời đi')

def check_input(file, file_train, x_inputs):
    print('Chờ kiểm tra đầu vào..........')
    file_ = [] 
    file_.append(file)
    test = data_process(file_)
    count = 0
    for t in test:
        for tr, f_tr in zip(x_inputs, file_train):
            if t == tr:
                count = count + 1
                print(t, tr)
                print(file, f_tr)
    if count == 0:
        print("Ok")
    else:
        print("File này thuộc dữ liệu test, hãy chọn lại!")
    print('Hoàn thành!')
    print()

def Selection(file_train, x_inputs, w_weights, bias):
    try:
        selection = input('Nhập lựa chọn của bạn: ')
                
        if selection == '1' or selection == '0':
            if selection == '1':
                file = input('Hãy chọn file cần nhận dạng: ')
                check_input(file, file_train, x_inputs)
                TEST(file, w_weights, bias)
                Selection(file_train, x_inputs, w_weights, bias)
            if selection == '0':
                print('OKEEE good bye ^_^')
                exit()
        else:
            print('Đọc kĩ và thử lại đi ~~~')
            Selection(file_train, x_inputs, w_weights, bias)
    except FileNotFoundError:
        print('Không tìm được file cần nhận dạng, hãy kiểm tra lại')
        Selection(file_train, x_inputs, w_weights, bias)  
#thuat toan lan truyen thang
if __name__ == "__main__":
    file_train = ['ga1.wav', 'ga2.wav', 'ga3.wav', 'ga4.wav', 'ga5.wav',
                  'ga6.wav', 'ga7.wav', 'ga8.wav', 'ga9.wav', 'ga10.wav',
                  'vit1.wav', 'vit2.wav', 'vit3.wav', 'vit4.wav', 'vit5.wav',
                  'vit6.wav', 'vit7.wav', 'vit8.wav', 'vit9.wav', 'vit10.wav']
    target = [1, 1, 1, 1, 1,
              1, 1, 1, 1, 1,
              0, 0, 0, 0, 0,
              0, 0, 0, 0, 0]
    epoch = 1000000000000
    print("Chờ chút để xử lý ..........")
    x_inputs = data_process(file_train)
    print("x_inputs = ", x_inputs)
    print("Ok xong một nửa")
    print()
    w_weights = [0.2, 0.4, -0.5, 1]
    bias = 0.7
    t1 = time.time()
    print('Chờ chút để học đã mới trả bài được nhé ^_^ .............')
    for i in range(0, epoch + 1):
        Y = []
        for x, t in zip(x_inputs, target):
            neural = Neural(x, w_weights, bias)
            output = neural.output()
            error = e(output, t)
            Y.append(output)
            if output != t:
                w_new = update_weight(w_weights, x, error)
                b_new = update_bias(bias, error)
                w_weights = w_new
                bias = b_new
        if (Y == target):
            t2 = time.time()
            print("Hoàn thành ở lần học thứ: ",i + 1)
            print(f"Hoàn thành trong {round((t2 - t1), 2)}s")
            print('Y = ',Y)
            print("Cập lại trọng số: ", w_weights)
            print("Cập nhạt bias: ", bias)
            print('Học xong rồi nha hí hí.')
            print()
            break
    print("=========*=============")
    print('chọn 1 để tiến hành nhận dạng, chọn 0 nếu muốn rời đi')
    Selection(file_train, x_inputs, w_weights, bias) 
    
    