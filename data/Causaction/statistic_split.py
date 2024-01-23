import json 
import random


with open('breakfast2.json',encoding='utf-8') as f: 
    all_data=json.load(f)
    
print(len(all_data))
all_data_filter=[]
for i in range(len(all_data)):
    if all_data[i]['segment_length']>=4:
        all_data_filter.append(all_data[i])
        
print(len(all_data_filter))

for j in range(5):   
    train_data=random.sample(all_data_filter,800)
    rest_data=[i for i in all_data_filter if i not in train_data]
    valid_data=rest_data[:100]
    test_data=rest_data[100:]

    print(len(train_data),len(valid_data),len(test_data))
    with open('fold{}/train.json'.format(j),'w',encoding='utf-8') as train:
        json.dump(train_data,train,ensure_ascii=False,sort_keys=True)
    with open('fold{}/valid.json'.format(j),'w',encoding='utf-8') as valid:
        json.dump(valid_data,valid,ensure_ascii=False,sort_keys=True)
    with open('fold{}/test.json'.format(j),'w',encoding='utf-8') as test:
        json.dump(test_data,test,ensure_ascii=False,sort_keys=True)    
