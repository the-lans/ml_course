import numpy as np
#import math as mt
#import matplotlib.pyplot as plt
#import time
from numpy import newaxis

#Загрузка CSV файла
def load_indata(filename):
    #Чтение файла
    fstr = open(filename, 'r').read()

    #Формирование двумерной таблицы данных
    in_data = []
    strvec = []
    strok = fstr.split('\n')
    for ind in range(len(strok)-1):
        strvec = strok[ind].split(';')
        for jnd in range(len(strvec)):
            strvec[jnd] = strvec[jnd].strip()
        in_data.append(strvec)
    return in_data

def load_data(filename, out_column):
    #Чтение файла
    fstr = open(filename, 'r').read()

    #Формирование двумерной таблицы данных
    in_data = []
    out_data = []
    strvec = []
    strok = fstr.split('\n')
    for ind in range(len(strok)-1):
        strvec = strok[ind].split(';')
        for jnd in range(len(strvec)):
            strvec[jnd] = strvec[jnd].strip()
        nvec = len(strvec) - out_column
        in_data.append(strvec[:nvec])
        out_data.append(strvec[nvec:])
    return [in_data, out_data]

#Сохранение CSV файла
def save_data(filename, data):
    fstr = open(filename, 'w')
    for datarow in data:
        newrow = True
        for datacolumn in datarow:
            if newrow != True: 
                fstr.write('; ')
            fstr.write(str(datacolumn))
            newrow = False
        fstr.write('\n')
    fstr.close()

#Транспонирование таблицы
def trans(matrix):
    matrix_t = list(zip(*matrix))
    return matrix_t
    
#Формирование последовательности
def calc_sequence(data, seq_len):
    result = []
    for ind in range(len(data) - seq_len + 1):
        dataseq = []
        for jnd in range(ind, ind + seq_len):
            dataseq.append(data[jnd])
        result.append(dataseq)
    return result

#Нормализация последовательности
def normalise_sequence(data):
    normalised_data = []
    for window in data:
        normalised_window = []
        base = window[0]
        for element in window:
            normalised_element = []
            for ind in range(len(base)):
                normalised_element.append(float(element[ind]) / float(base[ind]) - 1)
            normalised_window.append(normalised_element)
        normalised_data.append(normalised_window)
    return normalised_data

#Нормализация
def normalise(data):
    normalised_data = []
    for ind in range(len(data)):
        datarow_pref = float(data[ind])
        datarow_cur = float(data[ind + 1])
        normalised_row = []
        for jnd in range(len(datarow_pref)):
            normalised_row.append((datarow_pref[jnd] / datarow_cur[jnd]) - 1)
        normalised_data.append(normalised_row)
    return normalised_data

#Расчёт тестовых примеров
def predict(model, data_seq):
    #predicted_out = model.predict(data_seq)
    #print(predicted_out)
    predicted_out = []
    for ind in range(0, len(data_seq)):
        data_exm = data_seq[ind]
        dVec = model.predict(data_exm[newaxis,:,:])
        predicted_out.append(dVec[0,0])
    return predicted_out

#Из двумерного массива в одномерный
def matrix_to_array(matrix, jnd):
    array = []
    for ind in range(0, len(matrix)):
        array.append(float(matrix[ind][jnd]))
    return array

#Из входа получаем выход со смещением
def calc3D_out(data, out_column):
    in_data = []
    out_data = []
    for window in data:
        sep = len(window) - out_column
        in_datavec = window[:sep]
        out_datavec = window[sep:]
        in_data.append(in_datavec)
        out_vec = []
        for ind in range(out_column):
            out_vec.append(out_datavec[ind][0])
        out_data.append(out_vec)
    return [in_data, out_data]

#Разделение обучающего и тестового множества
def split_train_test(in_data, out_data, test_count, isShuffle = True):
    arrcount = [ind for ind in range(len(in_data))]
    arrcount = np.array(arrcount)
    if(isShuffle):
        np.random.shuffle(arrcount)
    
    in_result = []
    out_result = []
    for ind in range(len(arrcount)):
        indcount = arrcount[ind]
        in_result.append(in_data[indcount])
        out_result.append(out_data[indcount])

    in_result = np.array(in_result)
    out_result = np.array(out_result)
    #print(in_result.shape, out_result.shape)
    row = len(arrcount) - test_count
    in_train = in_result[:row]
    in_test = in_result[row:]
    out_train = out_result[:row]
    out_test = out_result[row:]
    #print(in_train.shape)

    return [in_train, in_test, out_train, out_test]

def calcdata3(data, N, column):
    data_ext = np.zeros((N, column, 1))
    data_ext[:, :, 0] = data.values
    return data_ext

def calcdata2(data, N):
    data_ext = np.zeros((N, 1))
    data_ext[:, 0] = data.values
    return data_ext
