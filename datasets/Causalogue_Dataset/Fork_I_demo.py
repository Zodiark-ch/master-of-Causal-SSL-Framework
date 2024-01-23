import openai 
openai.api_key=''
import system_all_2
import random
import time 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="Chain", type= str, help='')
parser.add_argument('--id', default="-1", type= int, help='')
parser.add_argument('--filename', default="dataset/", type= str, help='')
parser.add_argument('--system', default="1", type= str, help='')
parser.add_argument('--first_utterance_default', default="May I help you, sir?", type= str, help='')
parser.add_argument('--system_speaker1', default="You are a waiter in the restaurant that wants more fee.", type= str, help='')
parser.add_argument('--system_speaker2', default="You are a customer in the restaurant that does not want to tip the waiter. Because the water spilled red wine on you.", type= str, help='')
args = parser.parse_args()
print(args)





tem=1
pre_pen=2


    
def Fork_I_model(model,id,filename):
    
    while(True): 
        random_first_sen=random.choice(system_all_2.first_sentence)
        sreply1=random.choice(system_all_2.reply1)
        sreply2=random.choice(system_all_2.reply2)
        sreply3=random.choice(system_all_2.reply3)
        if sreply1!=sreply2 and sreply1!=sreply3 and sreply2!=sreply3:
            break;
    
    print('Speaker1:'+random_first_sen)
    text=str(id)+'\n'+'Fork_I'+'\n'
    text=text+'1.'+random_first_sen+'\n'
    #speaker2content
    speaker2 =openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": sreply1},
                {"role": "user","content": random_first_sen}
            ],
        temperature=tem,
        #max_tokens=128,
        stop=["\n","."],
        presence_penalty=pre_pen)
    print('Speaker2:'+speaker2.choices[0].message['content'])
    reply_1=speaker2.choices[0].message['content']
    text=text+'2.'+reply_1+'\n'
    speaker1 =openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": sreply2},
                {"role": "assistant", "content": random_first_sen},
            ],
        temperature=tem,
        #max_tokens=128,
        stop=["\n","."],
        presence_penalty=pre_pen)
    print('Speaker1:'+speaker1.choices[0].message['content'])
    reply_2=speaker1.choices[0].message['content']
    text=text+'3.'+reply_2+'\n'
    speaker2 =openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": sreply3},
                {"role": "user", "content": random_first_sen}
            ],
        temperature=tem,
        #max_tokens=128,
        stop=["\n","."],
        presence_penalty=pre_pen)
    print('Speaker2:'+speaker2.choices[0].message['content'])
    reply_3=speaker2.choices[0].message['content']
    text=text+'4.'+reply_3+'\n'
    text=text+'please label the dialogue, 1 represents positive sample, 0 represents other Fork samples,\n your label is:\n\n'
    
    print(' ---------------------The sample finished\n-----------------------')
    with open(filename,'a',encoding='utf-8') as f:
        f.write(text)
        f.close()
    
if __name__ == '__main__':
    Fork_I_model(args.model,args.id,args.filename)   
    