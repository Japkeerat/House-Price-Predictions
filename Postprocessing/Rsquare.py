"""
This class finds the adjusted R squared value.
"""

def average(y):
    sum_all = 0
    for num in y:
        sum_all = sum_all + num
    count = len(y)
    avg = sum_all / count
    return avg


def R_square(y_actual, y_predict):
    y_avg = average(y_actual)
    SS_res = 0
    SS_tot = 0
    Rsq = 0
    size = len(y_actual)
    for i in range(0, size):
        temp = y_actual[i] - y_predict[i]
        temp = temp * temp
        SS_res = SS_res + temp
        temp = y_actual[i] - y_avg
        temp = temp * temp
        SS_tot = SS_tot + temp
    try:
        Rsq = SS_res / SS_tot
    except:
        print("Divide by 0 encountered")
    Rsq = 1 - Rsq
    return Rsq


def adj_R_square(y_actual, y_predict, p, n):
    Rsq = R_square(y_actual, y_predict)
    adj_Rsq = 1 - Rsq
    try:
        adj_Rsq = adj_Rsq * ((n - 1) / (n - p - 1))
    except:
        print("Divide by 0 encountered")
    adj_Rsq = 1 - adj_Rsq
    return adj_Rsq