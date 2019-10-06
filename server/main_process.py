import sys
import os
import time


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


import socket
import pickle
import numpy as np
import struct 
import zlib

import cv2
import imutils
from imutils.video import VideoStream

from firebase import firebase

import smtplib
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.text import MIMEText
from email.mime.image import MIMEImage




#讀取模型設定
useNeuralStick=False
useTensor=False
face_detection_model="frozen_inference_graph.pb"
frozen_inference_graph="graph.pbtxt"


#註冊時總共取多少張圖片及隔多少秒取一張
num_extract_face=30
waitingSec_extract=0.3

#firebase
firebase = firebase.FirebaseApplication('https://ntnu-skyeyes.firebaseio.com/', authentication=None) 
num_registered=firebase.get('/trainingData/num_registered_host','')




class socket_receive_webcam(QThread):
        webcam_img=pyqtSignal(object)

        def __init__(self,_serverGUI):
                QThread.__init__(self)
                self.parent_serverGUI=_serverGUI
                
        def run(self):
                self.port=int(self.parent_serverGUI.IP_port_textLine.text())
                self.host=self.parent_serverGUI.host

                s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                print('Socket created')
                print("IP:"+self.host+"  Port:"+str(self.port))

                s.bind((self.host,self.port))
                print('Socket bind complete')
                s.listen(10)
                print('Socket now listening')

                conn,addr=s.accept()

                data = b""
                payload_size = struct.calcsize(">L")
                print("payload_size: {}".format(payload_size))
                while True:
                        conn.sendall(bytes(str(num_registered), 'utf-8'))  #傳server已註冊人數 給client 讓client知道自己的模型是否是最新的
                        _name=conn.recv(1024)
                        name_list=list(str(_name))
                        name=""
                        for i in name_list:
                                if i!='b' and i!="'":
                                        name+=i


                        while len(data) < payload_size:
                                #print("Recv: {}".format(len(data)))
                                data += conn.recv(4096)

                        #print("Done Recv: {}".format(len(data)))
                        packed_msg_size = data[:payload_size]
                        data = data[payload_size:]
                        msg_size = struct.unpack(">L", packed_msg_size)[0]
                        #print("msg_size: {}".format(msg_size))
                        while len(data) < msg_size:
                                data += conn.recv(4096)
                        frame_data = data[:msg_size]
                        data = data[msg_size:]

                        frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
                        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                        self.webcam_img.emit(frame)

                        #將辨識資訊儲存到server的detection_people array
                        if(name!="not"):

                                #若該人物 已在同個地點 重複出現 則更新 該人物被辨識到的畫面跟被辨識到的時間
                                recentName=0
                                localtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                totaltime=time.time()
                                for i in self.parent_serverGUI.detection_people:
                                        if name==i["name"]:
                                                i["totaltime"]=totaltime
                                                i["time"]=localtime
                                                i["img"]=frame
                                                recentName=1
                                                break
                                #若是第一次出現則 將辨識資訊儲存到server的detection_people array
                                if recentName==0:
                                        location=self.parent_serverGUI.location_textLine.text()
                                        detection_info={"name":name,"time":localtime,"loc":location,"img":frame,"totaltime":totaltime}
                                        self.parent_serverGUI.detection_people.append(detection_info)

                        
                                

                

class serverGUI(QWidget):
        def __init__(self,detectionGUI):
                super(QWidget,self).__init__()
                
                self.setupUi()
                self.socket_webcam=socket_receive_webcam(self)
                self.socket_webcam.webcam_img.connect(self.webcamEvent)
                self.parent_detectionGUI=detectionGUI
                self.host=""
                self.detection_people=[]
                

        def setupUi(self):
                location_label=QLabel()
                location_label.setFont(QFont("Timers" , 14 ,  QFont.Bold))
                location_label.setAlignment(Qt.AlignCenter)
                location_label.setText("監控地點:")

                self.location_textLine = QLineEdit()
                self.location_textLine.setFont(QFont("Timers" , 12 ,  QFont.Bold))
                self.location_textLine.setText("")
                self.location_textLine.setFixedSize(300,25)

                self.connected_btn=QPushButton()
                self.connected_btn.setText('連線')
                self.connected_btn.setFont(QFont("Timers" , 12 ,  QFont.Bold))
                self.connected_btn.setFixedSize(100,25)
                self.connected_btn.clicked.connect(self.connected_btn_clicked)
               
                location_layout=QHBoxLayout()
                location_layout.addWidget(location_label)
                location_layout.addWidget(self.location_textLine)
                location_layout.addWidget(self.connected_btn)



                IP_port_label=QLabel()
                IP_port_label.setFont(QFont("Timers" , 14 ,  QFont.Bold))
                IP_port_label.setAlignment(Qt.AlignCenter)
                IP_port_label.setText("Port:")


                self.IP_port_textLine = QLineEdit()
                self.IP_port_textLine.setFont(QFont("Timers" , 12 ,  QFont.Bold))
                self.IP_port_textLine.setText("")
                self.IP_port_textLine.setFixedSize(300,25)

                self.puase_btn=QPushButton()
                self.puase_btn.setText('暫停')
                self.puase_btn.setFont(QFont("Timers" , 12 ,  QFont.Bold))
                self.puase_btn.setFixedSize(100,25)
                #self.puase_btn.clicked.connect(self.puase_btn_btn_clicked)




                IP_port_layout=QHBoxLayout()
                IP_port_layout.addWidget(IP_port_label)
                IP_port_layout.addWidget(self.IP_port_textLine)
                IP_port_layout.addWidget(self.puase_btn)


                setting_layout=QVBoxLayout()
                setting_layout.addLayout(location_layout)
                setting_layout.addLayout(IP_port_layout)


                self.webcam = QLabel()
                self.webcam.setFixedHeight(400)
                self.webcam.setFixedWidth(500)
                img = cv2.imdecode(np.fromfile("art/disconnected.png", dtype=np.uint8), -1)
                self.webcam.setPixmap(self.getPixmap(img))
                


                V_Layout = QVBoxLayout()
                V_Layout.addLayout(setting_layout)
                V_Layout.addWidget(self.webcam)

                self.setLayout(V_Layout)

        def connected_btn_clicked(self):
                self.host=self.parent_detectionGUI.IP_textLine.text()
                self.socket_webcam.start()

        def puase_btn_btn_clicked(self):
                print(self.detection_people)
                print("暫停")


        def webcamEvent(self,frame):
                self.webcam.setPixmap(self.getPixmap(frame))

        def getPixmap(self,frame):
                img=cv2.resize(frame,(500,400))
                height, width, bytesPerComponent = img.shape
                bytesPerLine = 3 * width
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(QImg)
                return pixmap

        


class GUIcontroler(QWidget):

        registeredFlag = pyqtSignal(int)

        def __init__(self):
                super(GUIcontroler, self).__init__()
                
                self.setWindowTitle("NTNU-VIPlab 人臉辨識監控系統")
                
                self.startGUI = QWidget()
                self.detectionGUI = QWidget()
                self.registeredGUI = QWidget()
                self.recordGUI = QWidget()
		
                self.init_startGUI()
                self.init_detectionGUI()
                self.init_registeredGUI()
                self.init_recordGUI()

                self.GUIstack=QStackedWidget(self)
                self.GUIstack.addWidget(self.startGUI)
                self.GUIstack.addWidget(self.detectionGUI)
                self.GUIstack.addWidget(self.registeredGUI) 
                self.GUIstack.addWidget(self.recordGUI)

                self.timer1=timer(self)
                self.timer1.registered_webcam_img.connect(self.registeredGUI_webcamEvent)
                self.timer1.start()

                self.GUIstack_display(0)


        def GUIstack_display(self,i):
                self.GUIstack.setCurrentIndex(i)


        ######################################################################################
        #目錄GUI
        ######################################################################################
        def init_startGUI(self):
                label=QLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setFont(QFont("Timers" , 28 ,  QFont.Bold))
                label.setText("NTNU-VIPlab 人臉辨識監控系統")

                detectionBtn=QPushButton()
                detectionBtn.setFont(QFont("Timers" , 40 ,  QFont.Bold))
                detectionBtn.setText("監控畫面")
                detectionBtn.clicked.connect(self.startGUI_detectionBtn_clicked)
                detectionBtn.setFixedSize(400,200)

                registeredBtn=QPushButton()
                registeredBtn.setFont(QFont("Timers" , 40 ,  QFont.Bold))
                registeredBtn.setText("註冊")
                registeredBtn.clicked.connect(self.startGUI_registeredBtn_clicked)
                registeredBtn.setFixedSize(400,200)


                recordBtn=QPushButton()
                recordBtn.setFont(QFont("Timers" , 40 ,  QFont.Bold))
                recordBtn.setText("偵測紀錄")
                recordBtn.clicked.connect(self.startGUI_recordBtn_clicked)
                recordBtn.setFixedSize(400,200)



                layout = QGridLayout()
                layout.addWidget(label,0,1)
                layout.addWidget(registeredBtn,1,0,Qt.AlignTop)
                layout.addWidget(detectionBtn,1,1,Qt.AlignTop)
                layout.addWidget(recordBtn,1,2,Qt.AlignTop)
        
                self.startGUI.setLayout(layout)

        def startGUI_detectionBtn_clicked(self):
                self.GUIstack_display(1)
                

        def startGUI_registeredBtn_clicked(self):
                self.registeredFlag.emit(2)
                self.GUIstack_display(2)

        def startGUI_recordBtn_clicked(self):
                self.recordGUI_update_recordTable()
                self.GUIstack_display(3)                



        ######################################################################################
        #監控GUI
        ######################################################################################
        def init_detectionGUI(self):
                IP_label=QLabel()
                IP_label.setAlignment(Qt.AlignCenter)
                IP_label.setFont(QFont("Timers" , 20 ,  QFont.Bold))
                IP_label.setText("server IP:")

                self.IP_textLine = QLineEdit()
                self.IP_textLine.setFont(QFont("Timers" , 20 ,  QFont.Bold))
                self.IP_textLine.setText("")
                self.IP_textLine.setFixedSize(350,30)

                IP_setting_layout=QHBoxLayout()
                IP_setting_layout.addWidget(IP_label)
                IP_setting_layout.addWidget(self.IP_textLine)

                eamil_label=QLabel()
                eamil_label.setAlignment(Qt.AlignCenter)
                eamil_label.setFont(QFont("Timers" , 20 ,  QFont.Bold))
                eamil_label.setText("寄送通知信箱:")

                self.eamil_textLine = QLineEdit()
                self.eamil_textLine.setFont(QFont("Timers" , 20 ,  QFont.Bold))
                self.eamil_textLine.setText("")
                self.eamil_textLine.setFixedSize(350,30)

                eamil_setting_layout=QHBoxLayout()
                eamil_setting_layout.addWidget(eamil_label)
                eamil_setting_layout.addWidget(self.eamil_textLine)        

                IP_email_layout=QVBoxLayout()    
                IP_email_layout.addLayout(IP_setting_layout)
                IP_email_layout.addLayout(eamil_setting_layout)



                detection_bad_label=QLabel()
                detection_bad_label.setAlignment(Qt.AlignCenter)
                detection_bad_label.setFont(QFont("Timers" , 20 ,  QFont.Bold))
                detection_bad_label.setText("偵測危險人員:")

                self.detection_bad_textLine = QLineEdit()
                self.detection_bad_textLine.setFont(QFont("Timers" , 20 ,  QFont.Bold))
                self.detection_bad_textLine.setText("")
                self.detection_bad_textLine.setFixedSize(350,30)

                detection_bad_layout=QHBoxLayout()
                detection_bad_layout.addWidget(detection_bad_label)
                detection_bad_layout.addWidget(self.detection_bad_textLine)


                detection_good_label=QLabel()
                detection_good_label.setAlignment(Qt.AlignCenter)
                detection_good_label.setFont(QFont("Timers" , 20 ,  QFont.Bold))
                detection_good_label.setText("不需警告人員:")

                self.detection_good_textLine = QLineEdit()
                self.detection_good_textLine.setFont(QFont("Timers" , 20 ,  QFont.Bold))
                self.detection_good_textLine.setText("")
                self.detection_good_textLine.setFixedSize(350,30)

                detection_good_layout=QHBoxLayout()
                detection_good_layout.addWidget(detection_good_label)
                detection_good_layout.addWidget(self.detection_good_textLine)    

                detection_layout=QVBoxLayout()
                detection_layout.addLayout(detection_bad_layout)      
                detection_layout.addLayout(detection_good_layout)      
                



                restartBtn=QPushButton()
                restartBtn.setFont(QFont("Timers" , 12 ,  QFont.Bold,))
                restartBtn.setText("返回")
                restartBtn.clicked.connect(self.detectionGUI_restartBtn_clicked)
                restartBtn.setFixedSize(50,50)

                self.sys_broadcast_label=QLabel()
                self.sys_broadcast_label.setAlignment(Qt.AlignCenter)
                self.sys_broadcast_label.setFont(QFont("Timers" , 20 ,  QFont.Bold))
                pe = QPalette()
                pe.setColor(QPalette.WindowText,Qt.red)
                self.sys_broadcast_label.setPalette(pe)
                self.sys_broadcast_label.setText("尚未偵測到危險人物")

                broadcast_layout=QHBoxLayout()
                broadcast_layout.addWidget(restartBtn)
                broadcast_layout.addWidget(self.sys_broadcast_label)


              

                layout = QGridLayout()
                layout.addLayout(broadcast_layout,0,0)
                layout.addLayout(IP_email_layout,0,1,Qt.AlignCenter)
                layout.addLayout(detection_layout,0,2,Qt.AlignCenter)



                self.server1=serverGUI(self)
                layout.addWidget(self.server1,1,0,Qt.AlignCenter)

                self.server2=serverGUI(self)
                layout.addWidget(self.server2,1,1,Qt.AlignCenter)

                self.server3=serverGUI(self)
                layout.addWidget(self.server3,1,2,Qt.AlignCenter)


                self.server4=serverGUI(self)
                layout.addWidget(self.server4,2,0,Qt.AlignCenter)

                self.server5=serverGUI(self)
                layout.addWidget(self.server5,2,1,Qt.AlignCenter)

                self.server6=serverGUI(self)
                layout.addWidget(self.server6,2,2,Qt.AlignCenter)
        
                self.detectionGUI.setLayout(layout)

        def detectionGUI_restartBtn_clicked(self):
                self.GUIstack_display(0)
                


        ######################################################################################
        #註冊GUI
        ######################################################################################
       
        def init_registeredGUI(self):
                self.registeredGUI_L1=QLabel()
                self.registeredGUI_L1.setFont(QFont("Timers" , 28 ,  QFont.Bold))
                self.registeredGUI_L1.setAlignment(Qt.AlignCenter)
                self.registeredGUI_L1.setText("輸入使用者名稱:")

                dummyLabel=QLabel()
                dummyLabel.setFont(QFont("Timers" , 20 ,  QFont.Bold))
                dummyLabel.setAlignment(Qt.AlignCenter)
                dummyLabel.setText("    ")
          
                self.registeredGUI_textLine = QLineEdit()
                self.registeredGUI_textLine.setFont(QFont("Timers" , 28 ,  QFont.Bold))
                self.registeredGUI_textLine.setText("unknown")
                self.registeredGUI_textLine.setFixedSize(500,50)

                self.registeredGUI_sureBtn = QPushButton()
                self.registeredGUI_sureBtn.setFont(QFont("Timers" , 28 ,  QFont.Bold))
                self.registeredGUI_sureBtn.setText("確定")
                self.registeredGUI_sureBtn.clicked.connect(self.registeredGUI_sureBtn_clicked)
                self.registeredGUI_sureBtn.setFixedSize(200,100)

                self.registeredGUI_restartBtn=QPushButton()
                self.registeredGUI_restartBtn.setFont(QFont("Timers" , 16 ,  QFont.Bold))
                self.registeredGUI_restartBtn.setText("返回上一頁")
                self.registeredGUI_restartBtn.clicked.connect(self.registeredGUI_restartBtn_clicked)
                self.registeredGUI_restartBtn.setFixedSize(150,100)

                self.registeredGUI_webcam = QLabel()



                mid_layout=QVBoxLayout()
                mid_layout.addWidget(self.registeredGUI_L1)
                mid_layout.addWidget(self.registeredGUI_textLine)
                mid_layout.addWidget(dummyLabel)
                mid_layout.addWidget(self.registeredGUI_webcam)


                

                self.registeredGUI_layout = QGridLayout()
                #把個物件放到九宮格內並在各格內置中放置
                self.registeredGUI_layout.addWidget(self.registeredGUI_restartBtn,0,0,Qt.AlignLeft)
                self.registeredGUI_layout.addLayout(mid_layout,1,0,Qt.AlignCenter)
                self.registeredGUI_layout.addWidget(self.registeredGUI_sureBtn,2,0,Qt.AlignCenter)
              

                self.registeredGUI.setLayout(self.registeredGUI_layout)

        def registeredGUI_restartBtn_clicked(self):
                self.registeredGUI_L1.setText("輸入使用者名稱:")
                self.registeredGUI_textLine.setText("unknown")
                self.registeredFlag.emit(0)
                self.GUIstack_display(0)
               
        def registeredGUI_sureBtn_clicked(self):
                if self.registeredGUI_textLine.text()=="unknown":
                        self.registeredGUI_L1.setText("您還沒輸入您的姓名")
                else:
                        self.registeredFlag.emit(1)

        def registeredGUI_webcamEvent(self,frame):
                self.registeredGUI_webcam.setPixmap(self.getPixmap(frame))

        def getPixmap(self,frame):
                img=cv2.resize(frame,(500,400))
                height, width, bytesPerComponent = img.shape
                bytesPerLine = 3 * width
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(QImg)
                return pixmap

        ######################################################################################
        #偵測紀錄GUI
        ######################################################################################
        def init_recordGUI(self):

                restartBtn=QPushButton()
                restartBtn.setFont(QFont("Timers" , 12 ,  QFont.Bold,))
                restartBtn.setText("返回")
                restartBtn.clicked.connect(self.recordGUI_restartBtn_clicked)
                restartBtn.setFixedSize(50,50)



                self.recordTable=QTableWidget()
                self.recordTable.setFixedWidth(1800)
                self.recordTable.setFont(QFont("Timers" , 24 ,  QFont.Bold))
                #self.recordTable.setFrameShape(QFrame.NoFrame)  ##设置无表格的外框
                self.recordTable.setEditTriggers(QAbstractItemView.NoEditTriggers) #设置表格不可更改
                self.recordTable.setSelectionMode(QAbstractItemView.NoSelection)#不能选择
                self.recordTable.verticalHeader().setVisible(False)  #隱藏垂直的標頭 也就是col 0 的元素都隱藏
                self.recordTable.horizontalHeader().setStretchLastSection(True)#设置第五列宽度自动调整，充满屏幕
                #self.recordTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)#自動等宽
                
                
                self.recordTable.setColumnCount(4)
                self.recordTable.setHorizontalHeaderLabels(["人名","地點","時間","辨識畫面"])

                #self.orderingGUI_chargeTable.setItem(index,1, QTableWidgetItem(""))

                layout = QGridLayout()
                layout.addWidget(restartBtn,0,0)
                layout.addWidget(self.recordTable,1,1)

       

        
                self.recordGUI.setLayout(layout)


        def recordGUI_update_recordTable(self):
                self.recordTable.setRowCount(len(self.timer1.detection_people))

                index=0
                for key in self.timer1.detection_people:
                        self.recordTable.setItem(index,0, QTableWidgetItem(key['name']))
                        self.recordTable.setItem(index,1, QTableWidgetItem(key['loc']))
                        self.recordTable.setItem(index,2, QTableWidgetItem(key['time']))

                        face_img_label=QLabel("")
                        face_img_label.setAlignment(Qt.AlignCenter)
                        face_img_label.setPixmap(self.getPixmap(key['img']))
                        self.recordTable.setCellWidget(index,3, face_img_label)

                        self.recordTable.setColumnWidth(0,200)
                        self.recordTable.setColumnWidth(1,200)
                        self.recordTable.setColumnWidth(2,200)
                        self.recordTable.setColumnWidth(3,200)
                        self.recordTable.setRowHeight(index,400)

                        index+=1                


        def recordGUI_restartBtn_clicked(self):

                for i in range(self.recordTable.rowCount()):
                        self.recordTable.removeRow(i)                
                self.GUIstack_display(0)
                

            
       

class timer(QThread):
        registered_webcam_img=pyqtSignal(object)

        def __init__(self,_GUIcontroler):
                QThread.__init__(self)
                #註冊端 變數設定
                self.parentProcess = _GUIcontroler
                self.registeredFlag = 0   #沒進入註冊GUI0 進入註冊GUI時2 按下註冊按鈕後 變成1
                self.parentProcess.registeredFlag.connect(self.set_registeredFlag)

                #儲存辨識到的人
                self.detection_people=[]

                self.gmail_user = 'ntnuviplab.teamfacenet@gmail.com'
                self.gmail_password = 'teamfacenet' # your gmail password     
                self.server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                self.server.ehlo()
                self.server.login(self.gmail_user, self.gmail_password)           



        def run(self):
                
                self.load_recognizeFaceModel()
                already_sendMail_people=[]

                while (True):



                        #把各伺服器偵測到的人臉資料合併成一list
                        detection_people1=self.parentProcess.server1.detection_people
                        detection_people2=self.parentProcess.server2.detection_people
                        detection_people3=self.parentProcess.server3.detection_people
                        detection_people4=self.parentProcess.server4.detection_people
                        detection_people5=self.parentProcess.server5.detection_people
                        detection_people6=self.parentProcess.server6.detection_people
                        self.detection_people=detection_people1+detection_people2+detection_people3+detection_people4+detection_people5+detection_people6
                        
                        #利用該list 發現危險人物print警告訊息到GUI及寄信處理
                        detection_person=None
                        for i in self.detection_people:
                             if  self.parentProcess.detection_bad_textLine.text()==i['name']:
                                     self.parentProcess.sys_broadcast_label.setText('警告!'+i['name']+'出現在'+i['loc']+'\n'+'於'+i['time'])
                                     detection_person=i
                                     break

                        if detection_person!=None:

                                #寄送信件 寄第一封後 若還是被偵測到 60秒內會再寄第二封
                                already_send=0#check被偵測到的該人物 是否被寄過信的flag
                                for j in already_sendMail_people:
                                     if detection_person['name']==j['name']:
                                             dif=detection_person['totaltime']-j['totaltime']
                                             if dif>60:
                                                     self.send_img_email(detection_person)
                                                     send_one_flag=1
                                                     j['totaltime']=detection_person['totaltime']
                                             already_send=1
                                             break
                                     
                                if already_send==0:
                                        self.send_img_email(detection_person)
                                        totaltime=time.time()
                                        send_info={"name":detection_person['name'],"totaltime":totaltime}
                                        already_sendMail_people.append(send_info)

                                    

                                
                                             



                                
                                                      
                        
                       #註冊相關問題處理
                       #registeredFlag=0代表不在註冊GUI  registeredFlag=1在註冊GUI按下註冊按鈕 registeredFlag=2進入註冊GUI
                        if self.registeredFlag==2:                   #進入註冊GUI 啟用webcam
                                print("[INFO-註冊GUI]啟用攝像機")
                                vs = VideoStream(src=0).start()
                                while(True):
                                        frame = vs.read()
                                        frame = imutils.resize(frame, width=600)
                                        if self.registeredFlag==1:    #按下註冊按鈕 就註冊
                                                self.registered(vs)
                                        elif self.registeredFlag==2:  #只有在註冊GUI 才送webcam圖片到GUI
                                                self.registered_webcam_img.emit(frame)
                                        elif self.registeredFlag==0:
                                                vs.stream.release()
                                                print("[INFO-註冊GUI]釋放攝像機")
                                                break


        def sendEmail(self,badman_info):
                gmail_user = 'ntnuviplab.teamfacenet@gmail.com'
                gmail_password = 'teamfacenet' # your gmail password

                content="親愛的用戶您好:"+"\n於"+badman_info['time']+"監控系統在"+badman_info['loc']+"辨識到了"+badman_info['name']+"\n請您注意您的人身安全!"
                msg = MIMEText(content)
                msg['Subject'] = "[警告]"+badman_info['name']+"出現在"+badman_info['loc']
                msg['From'] = gmail_user
                msg['To'] = self.parentProcess.eamil_textLine.text()

                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                server.ehlo()
                server.login(gmail_user, gmail_password)
                server.send_message(msg)
                
                server.quit()

                print('Email sent!')


        def send_img_email(self,badman_info):


                # 下面依次為郵件類型，主題，發件人和收件人。
                msg = MIMEMultipart('mixed')
                msg['Subject'] = "[警告]"+badman_info['name']+"出現在"+badman_info['loc']
                msg['From'] = self.gmail_user
                msg['To'] = self.parentProcess.eamil_textLine.text()

                content="親愛的用戶您好:"+"\n於"+badman_info['time']+"監控系統在"+badman_info['loc']+"辨識到了"+badman_info['name']+"\n請您注意您的人身安全!"

                # 下面為郵件的正文
                text_plain = MIMEText(content, 'plain', 'utf-8')
                msg.attach(text_plain)
                
                # 構造圖片連結
                imgFileName=badman_info['name']+".jpg"
                cv2.imwrite('output/'+imgFileName, badman_info['img'])
                sendimagefile = open('output/'+imgFileName, 'rb').read()
                image = MIMEImage(sendimagefile)
                # 下面一句將收件人看到的附件照片名稱改為people.png。
                image["Content-Disposition"] = 'attachment; filename="people.png"'
                msg.attach(image)
                send_msg=msg.as_string()

                self.server.sendmail(self.gmail_user,self.parentProcess.eamil_textLine.text(),send_msg)
                print('Email sent!')



   #     def send_img_email(self,badman_info):
    #            gmail_user = 'ntnuviplab.teamfacenet@gmail.com'
     #           gmail_password = 'teamfacenet' # your gmail password

                # 下面依次為郵件類型，主題，發件人和收件人。
      #          msg = MIMEMultipart('mixed')
       #         msg['Subject'] = "[警告]"+badman_info['name']+"出現在"+badman_info['loc']
        #        msg['From'] = gmail_user
         #       msg['To'] = self.parentProcess.eamil_textLine.text()

          #      content="親愛的用戶您好:"+"\n於"+badman_info['time']+"監控系統在"+badman_info['loc']+"辨識到了"+badman_info['name']+"\n請您注意您的人身安全!"

                # 下面為郵件的正文
           #     text_plain = MIMEText(content, 'plain', 'utf-8')
            #    msg.attach(text_plain)
                
                # 構造圖片連結
             #   imgFileName=badman_info['name']+".jpg"
              #  cv2.imwrite('output/'+imgFileName, badman_info['img'])
              #  sendimagefile = open('output/'+imgFileName, 'rb').read()
              #  image = MIMEImage(sendimagefile)
                # 下面一句將收件人看到的附件照片名稱改為people.png。
              #  image["Content-Disposition"] = 'attachment; filename="people.png"'
              #  msg.attach(image)
              #  send_msg=msg.as_string()

               # server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
               # server.ehlo()
               # server.login(gmail_user, gmail_password)
               # server.sendmail(gmail_user,self.parentProcess.eamil_textLine.text(),send_msg)
               # server.quit()
               # print('Email sent!')


        def registered(self,vs):        
                #若該使用者還未註冊過 則建立資料夾
                registeredName=self.parentProcess.registeredGUI_textLine.text()
                
                print("[INFO]registered people:")
                data = firebase.get('/trainingData','')
                print(data["names"])
                print("Is "+registeredName+" in dataset? "+str(registeredName in data["names"]))

                if not (self.parentProcess.registeredGUI_textLine.text() in data["names"]):
                        self.parentProcess.registeredGUI_L1.setText("請上下左右搖晃您的臉頰")
                        i=0
                        dataset=[]
                        while i<num_extract_face:
                                frame = vs.read()
                                frame = imutils.resize(frame, width=600)
                                dataset.append(frame)
                                self.registered_webcam_img.emit(frame)
                                time.sleep(waitingSec_extract)
                                i += 1

                        self.registeredFlag=2

                        self.parentProcess.registeredGUI_L1.setText("使用者特徵萃取中...")
                        img = cv2.imread("art/training.jpg")  #loading圖片
                        self.registered_webcam_img.emit(img)

                        self.extract_embeddings(dataset,registeredName,data)

                        global num_registered
                        num_registered+=1
                        firebase.put('/trainingData','num_registered_host',num_registered)
                        self.parentProcess.registeredGUI_L1.setText("完成")
                        
                        
                self.registeredFlag=2


        def extract_embeddings(self,dataset,userName,data):
                print("[INFO]extract embeddings...")
                knownEmbeddings = []

                for iface in range(len(dataset)):
                        print("[INFO] processing image "+str(iface+1)+"/"+str(len(dataset)))

                        image=dataset[iface]
                        image = imutils.resize(image, width=600)
                        (h, w) = image.shape[:2]
                        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
                        
                        self.detector.setInput(imageBlob)
                        detections = self.detector.forward()
                        
                        if len(detections) > 0:   #確保圖片中至少一個臉存在

                                    #若有複數臉從中挑選信心值最大的臉
                                    i = np.argmax(detections[0, 0, :, 2])
                                    confidence = detections[0, 0, i, 2]
                                    
                                    filter_confidence = 0.5

                                    if confidence > filter_confidence:#若信心值最大的臉高於filter_confidence才萃取特徵向量
                                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                            (startX, startY, endX, endY) = box.astype("int")
                                            face = image[startY:endY, startX:endX]
                                            (fH, fW) = face.shape[:2]
                                            
                                            if fW < 20 or fH < 20:
                                                    continue
                                                    
                                            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                                            self.embedder.setInput(faceBlob)
                                            vec = self.embedder.forward()
                                            knownEmbeddings.append(vec.flatten())

                


                for i in knownEmbeddings:
                        data["embeddings"].append(i)
                        data["names"].append(userName)

                
                #存放names
                firebase.put('/trainingData','names',data['names'])


                #將float32轉成firebase能接受的float資料型態
                embedding=[]
                for i in range(len(data['embeddings'])):
                        embedding.append([])
                        for j in range(len(data['embeddings'][i])):
                                embedding[i].append(float(data['embeddings'][i][j]))
                
                self.parentProcess.registeredGUI_L1.setText("使用者特徵存入資料庫...")

                #才能把embedding存入
                for i in range(len(data['embeddings'])):
                        firebase.put('/trainingData/embeddings',str(i),embedding[i])
                
                print("[INFO]completed")

 
        def set_registeredFlag(self, data):
                self.registeredFlag=data


        def load_recognizeFaceModel(self):
                # load our serialized face detector from disk
                print("[INFO] loading face detector...")

                if useTensor:
                        protoPath = os.path.sep.join(["face_detection_model", frozen_inference_graph])
                        modelPath = os.path.sep.join(["face_detection_model",face_detection_model])
                        self.detector = cv2.dnn.readNetFromTensorflow(modelPath,protoPath)
                else:
                        protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
                        modelPath = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
                        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

                # load our serialized face embedding model from disk
                print("[INFO] loading face recognizer...")
                self.embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")


                if useNeuralStick:
                        print("[INFO]setting a neural stick is activated")
                        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
                        self.embedder.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                

                # load the actual face recognition model along with the label encoder
                self.recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
                self.le = pickle.loads(open("output/le.pickle", "rb").read())
                print("[INFO]completed")

       

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = GUIcontroler()
    MainWindow.showMaximized()

    sys.exit(app.exec_())
