"""
This class calculates the accuracy of the models.
"""

def get_accuracy(test, predict, handle_error):
    accurate_list = get_accurate_list(test, predict, handle_error)
    total = len(accurate_list)
    count_correct = accurate_list.count("1")
    accuracy = count_correct/total
    accuracy = accuracy*100
    return accuracy
    
    
def get_accurate_list(test, predict, handle_error):
    accurate_list = []
    size = len(test)
    for i in range(size):
        if(predict[i] <= (test[i]+(handle_error/100)*test[i]) and 
           predict[i] >= (test[i]-(handle_error/100)*test[i])):
            accurate_list.append("1")
        else:
            accurate_list.append("0")
    return accurate_list