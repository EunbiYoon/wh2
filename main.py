from utils import load_training_set, load_test_set
from itertools import chain
import pandas as pd

# unique vocabulary vector
def vocabulary_vector(data_list):
    return 3

# Prioir Probability :: pr(yi) = N(yi)/N
def prior_probability(pos_data_len, neg_data_len, train_info):
    if train_info=="pos":
        pr_yi = pos_data_len / ( pos_data_len + neg_data_len )
    elif train_info=="neg":
        pr_yi = neg_data_len / ( pos_data_len + neg_data_len )
    else:
        print("Error :: check your train_info")
    return pr_yi
        
# bag-of-words vector m
def bag_of_words(V_vector , data_list):
    M_Vector=pd.DataFrame(columns=V_vector,index=range(len(data_list)))
    print(M_Vector)

# Posterior Probability :: pr(yi|Doc) = n(wk, yi)/sigma (n(ws,yi))
def posterior_probability():
    return 4
                                                                                                                                                  

# make confusion matrix
def confusion_matrix():
    return 5



# main function
if __name__ == '__main__':
    ### preprocessign data
    percentage_positive_instances_train = 0.2
    percentage_negative_instances_train = 0.2
    percentage_positive_instances_test = 0.2
    percentage_negative_instances_test = 0.2
    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    ### calculate prior probability
    pr_yi_pos=prior_probability(len(pos_train), len(neg_train), "pos")
    pr_yi_neg=prior_probability(len(pos_train), len(neg_train), "neg")
    print(pr_yi_pos)
    print(pr_yi_neg)


    ### check data output
    # print("Number of positive training instances:", len(pos_train))
    # print("Number of positive training instances:", len(neg_train))
    # print("Number of positive test instances:", len(pos_test))
    # print("Number of negative test instances:", len(neg_test))
    # # 데이터 출력
    # print(pos_train)
    # print(neg_train)
    # print(pos_test)
    # print(neg_test)
    # 데이터 저장
    with open('pos_train.txt','w',encoding='utf-8') as file:
        file.writelines(f"{item}\n" for item in pos_train)
    
    ### combine pos and neg data 
    combined_train=list(chain(pos_train+neg_train))
    # 데이터 저장
    with open('combined.txt','w',encoding='utf-8') as file:
        file.writelines(f"{item}\n" for item in combined_train)

    ### calculate Vocabulary Vector : unique words from training
    V_vector = list(set(chain.from_iterable(combined_train)))

    ### calculate posterior probability
    aa=bag_of_words(V_vector,combined_train)
    print(aa)
