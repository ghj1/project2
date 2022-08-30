import cv2
import tkinter as tk
from PIL import Image, ImageTk,ImageOps
from keras.models import load_model
import numpy as np
from functools import partial
import speech_recognition as sr
from gtts import gTTS
import os
import time
from playsound import playsound


model = load_model('./model/eye_model.h5')
model2 = load_model('./model/nose_model.h5') # 모델 읽어오기
model3 = load_model('./model/lip_model.h5') 
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


IMAGE_PATH = './img/우주.jpg'
WIDTH, HEIGTH = 1000,700 

window = tk.Tk()
window.geometry('{}x{}'.format(WIDTH, HEIGTH))
window.title('《Honey》 『꿀』 관상')

canvas3 = tk.Canvas(window, width= WIDTH, height= HEIGTH)
canvas3.pack()
img5 = ImageTk.PhotoImage(Image.open(IMAGE_PATH).resize((WIDTH, HEIGTH), Image.ANTIALIAS))
canvas3.background = img5
bg = canvas3.create_image(0, 0, anchor=tk.NW, image=img5)


opencvFrame = tk.Frame(window)
opencvFrame.place(x=10,y=40)
lmain = tk.Label(opencvFrame)
lmain.grid(row=25, column=0)


canvas = tk.Canvas(window, width = 230, height = 250)
img = ImageTk.PhotoImage(Image.open("./img/profile.jpg"))
canvas.create_image(0, 0,anchor="nw",  image=img)
canvas.place(x=350,y = 40)


#canvas2 = tk.Canvas(window, width = 493, height = 200)
#img2 = ImageTk.PhotoImage(Image.open("./img/profile.jpg"))
#canvas2.create_image(0, 0,anchor="nw",  image=img2)
#canvas2.place(x=350,y =330)
# opencv 카메라 연결
cap = cv2.VideoCapture(0)
def show_frame():
    _, frame = cap.read()
    frame= frame[200:450,250:480]
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame,(300,400))
    # 이미지 처리
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame) # 10마이크로 초 이후 show_frame 다시 불러움
def capture(): # 현재 웹캠 정보를 이미지로 저장하는 함수입니다.
    _, frame = cap.read()
    img_frame = frame 
    img_size = img_frame[200:450,250:480]
    cv2.imwrite('result/screenshot.jpg', img_size)
    print('저장됨')
def printfacetype(i): #관상 프로그램을 출력하는 함수입니다.
    global img, canvas,canvas3
    if os.path.isfile('voice_0.mp3') ==True:
        os.remove('voice_0.mp3')
    
    images = np.array(Image.open(i))
    size = (224, 224)
    image = cv2.resize(images, size)
    image = Image.fromarray(image)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    prediction2 = model2.predict(data)
    prediction3 = model3.predict(data)
    img = ImageTk.PhotoImage(Image.open("result/screenshot.jpg"))
    canvas.create_image(0, 0,anchor="nw",  image=img)
    #canvas.place(x=350,y =10)
    #canvas = tk.Canvas(window, width = 493, height = 249)
    num = prediction.argmax()
    num2 = prediction2.argmax() #모델 파일과 비교하여 가장 높은 값을 추출함.
    num3 = prediction3.argmax()
    eye =['사자눈, 당신은 항상 앞으로 나아가는 성격입니다.\n호탕한 마음을 갖고 있지만 욕심을 낼수록 당신이 사랑하는 사람을 멀어지게 할 수 있습니다.\n','호랑이눈, 당신은 정의감이 넘치며 자존심이 강합니다. \n정의의 사도처럼 행동하고 싶지만 그럴수록 힘들어집니다.\n','원앙눈, 당신은 재물복이 좋으며 부부 관계가 좋을 것입니다.\n하지만 음란해지는 것을 경계하여야 합니다.\n','소눈, 당신은 인내심이 강하고 우직하면서도 인자한 성품을 지녔고\n타인에게 친절한 마음을 갖고 있습니다.\n','거북이눈, 정이 많고 신망이 두터워 신의를 져 버리지 않는 사람입니다.\n때로는 손해를 볼 수도 있지만, 사람들이 알아줄 날이 올겁니다.\n']
    msg = eye[num]
    
    nose = ['용코, 당신은 아름다운 삶을 살 것이고 후세에도 좋은 영향을 줄 것입니다. \n한 분야에서 성공할 수 있으니 하루하루 매순간 최선을 다하세요.\n','마늘코, 당신은 사람들에게 인정받고 존경 받을 수 있습니다.\n또한 형제 간의 우애가 좋습니다. 말년에 가장 행복한 인생을 살게 될 것입니다.\n','주머니코, 직장에서 당신은 아주 높은 직위까지 올라갑니다.\n당신의 일생에 돈 걱정이 가장 쓸데없는 걱정이 될 겁니다.\n','사자코, 당신은 성공을 위해 부단히 노력합니다.\n당신은 명성은 얻으나 돈이 새나갈 수 있으니 사치를 주의해야 합니다.\n','현담비, 당신은 노력하는 것에 비해 훨씬 더 많은 것을 얻을 것입니다. \n최선을 다한다면 부귀영화를 누릴 것입니다.\n']
    msg2= nose[num2]
    
    lip = ['사자구, 당신은 재주가 많고 총명합니다. \n당신은 공부를 하면 출세할 수 있습니다. ','앙월구, 당신은 직장 운이 매우 좋네요. \n공무원을 하는 것도 좋을 것 같습니다. ','만궁구, 당신은 부귀를 누리게 됩니다. \n특히 중년이 지나고는 돈이 저절로 들어오게 될 겁니다.','취화구, 당신은 앞날에 대한 걱정이 지나치게 많은 사람입니다. \n하지만 끈기를 갖고 한가지 일에 몰두하면 성공할 수 있습니다.','앵도구, 당신은 총명하며, 스마트합니다. \n타고난 능력을 바탕으로 사람들의 주목을 받을 겁니다.']
    
    msg3 = lip[num3]
    text=' 당신의 눈은 %s\n 당신의 코는 %s\n 당신의 입은 %s' %(msg,msg2,msg3)
    
    for i in range(1):
        
        tts= gTTS(text,lang='ko')
        filename = 'C:/Users/PC021/voice_%d.mp3'%(i)
        
        tts.save(filename)
        print('file saved')
        canvas3.create_text(350,320,anchor="nw",text=text,font =('맑은 고딕', 12),fill='white')    
        #playsound(filename,True)
        print('playing music')
        print(text)
        # canvas3.delete(id)
     
show_frame()
#font = tk.font.Font(family = "Gabriola", size = 17)
button_1 = tk.Button(window, text = "촬영", command = capture, bg= "SlateBlue2",
                                fg = "snow",font =('맑은 고딕', 20), width= 6, height= 1 ,bd= 0, relief = "ridge")
button_1.place(x = 20, y = 480) #button1은 캡처하는 버튼입니다.

button_2 = tk.Button(window,  text = "관상", command =partial(printfacetype,'result/screenshot.jpg'), backg = "gray19",
                                fg = "snow", font =('맑은 고딕', 20), width= 6, height= 1, bd = 0 ,relief = "ridge")
button_2.place(x = 210, y = 480) #button2는 관상프로그램 실행 버튼입니다.

window.mainloop()
