from firebase import firebase
import pickle



firebase = firebase.FirebaseApplication('https://ntnu-skyeyes.firebaseio.com/', authentication=None) 


data = pickle.loads(open("embeddings.pickle", "rb").read())

#先清空資料
firebase.delete('/trainingData','')

#存放names
firebase.put('/trainingData','names',data['names'])


#將float32轉成firebase能接受的float資料型態
embedding=[]
for i in range(len(data['embeddings'])):
    embedding.append([])
    for j in range(len(data['embeddings'][i])):
        embedding[i].append(float(data['embeddings'][i][j]))
#才能把embedding存入
for i in range(len(data['embeddings'])):
    firebase.put('/trainingData/embeddings',str(i),embedding[i])

#從firebase讀取vector方法
get_embedding=firebase.get('/trainingData','')



#初始化 local 現在已經註冊過的人 
names=list(set(data["names"]))
f = open("num_registered_local.pickle", "wb")
f.write(pickle.dumps(len(names)))
f.close()

#初始化 host 現在已經註冊過的人 
firebase.put('/trainingData','num_registered_host',len(names))


print("completed")
