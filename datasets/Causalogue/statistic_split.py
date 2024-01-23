import json 
import random


with open('all_data_small.json',encoding='utf-8') as f: 
    all_data=json.load(f)
    
print(len(all_data))


for j in range(10):   
    train_data=random.sample(all_data,1338)
    rest_data=[i for i in all_data if i not in train_data]
    valid_data=random.sample(rest_data,100)
    test_data=[i for i in rest_data if i not in valid_data]

    print(len(train_data),len(valid_data),len(test_data))
    with open('all_sample_small/fold{}/train.json'.format(j),'w',encoding='utf-8') as train:
        json.dump(train_data,train,ensure_ascii=False,sort_keys=True)
    with open('all_sample_small/fold{}/valid.json'.format(j),'w',encoding='utf-8') as valid:
        json.dump(valid_data,valid,ensure_ascii=False,sort_keys=True)
    with open('all_sample_small/fold{}/test.json'.format(j),'w',encoding='utf-8') as test:
        json.dump(test_data,test,ensure_ascii=False,sort_keys=True)    
