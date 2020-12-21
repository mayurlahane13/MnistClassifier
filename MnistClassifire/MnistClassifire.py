
from tkinter import *
#from PIL import Image, ImageDraw, ImageGrab
import math
import os
#from pyscreenshot import grab
import keras 
import numpy as np 
import csv 
import pandas as pd 
import tensorflowjs as tfjs



data = pd.read_csv('mnist_train.csv', header = 0)
#arr = np.array(data.iloc[0])
arr = np.array(data.iloc[0:,0])
inputs = np.array(data.iloc[:,1:785], dtype = float)

for i in range(len(inputs)):
	for j in range(len(inputs[0])):
		inputs[i][j] = inputs[i][j] / 255.0


outputs = np.zeros(shape=(59999, 10))
for i in range(len(arr)):
	output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	ele = arr[i]
	output[ele] = 1
	outputs[i] = output


'''model = keras.models.Sequential()
#1st hidden layer
model.add(keras.layers.Dense(64, input_shape=(784,)))
model.add(keras.layers.Activation('tanh'))
#2nd hidden layer
#model.add(keras.layers.Dense(300))
#model.add(keras.layers.Activation('tanh'))
#output layer
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))

optimizer = keras.optimizers.Adam(0.01)

model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

model.fit(inputs, outputs, nb_epoch = 10, shuffle = True)'''

model = keras.models.load_model('MnistModel.h5')


print(outputs[0])
prediction = model.predict(np.array([inputs[0]]))
print(np.argmax(prediction))
print(outputs[100])
prediction = model.predict(np.array([inputs[100]]))
print(np.argmax(prediction))
print(outputs[500])
prediction = model.predict(np.array([inputs[500]]))
print(np.argmax(prediction))

#model.save('MnistModel.h5')

tfjs.converters.save_keras_model(model, '/Users/makarandsubhashlahane/Desktop/Projects/JavaScript/Tensorflow.jsProjects/MnistClassifireTesting/tfMnistModel')



# Draw on the screen a green line from (0, 0) to (100, 100)
# that is 5 pixels wide.
#pygame.draw.line(screen, GREEN, [0, 0], [100, 100], 5)

'''pygame.init()
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
width = 700
height = 700
window = pygame.display.set_mode((width, height))
canvas = window.copy()
pygame.display.set_caption("Mnist Classifire")
#loop until the user clicks the close button
done = False
# Used to manage how fast the screen updates
clock = pygame.time.Clock()
# -------- Main Program Loop -----------

while not done:
	#--Main event loop 
	for event in pygame.event.get(): #User did something
		if event.type == pygame.QUIT: #if user clicked close
			done = True
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_c:
				window.fill(BLACK)
				canvas.fill(BLACK)


	

	#pygame.draw.rect(screen, RED, [55, 50, 20, 25], 0)
	left_pressed, middle_pressed, right_pressed = pygame.mouse.get_pressed()
	if left_pressed:
		pygame.draw.circle(window, WHITE, (pygame.mouse.get_pos()), 40)
	# --- Go ahead and update the screen with what we've drawn.
	pygame.display.update()

    # --- Limit to 60 frames per second
	clock.tick(60)

pygame.quit()

from tkinter import *

canvas_width = 700
canvas_height = 700

image1 = Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

imageFilename = 'myNumber.jpg'

def paint( event ):
   python_green = "#476042"
   x1, y1 = ( event.x - 30 ), ( event.y - 30 )
   x2, y2 = ( event.x + 30 ), ( event.y + 30 )
   w.create_oval( x1, y1, x2, y2, fill = 'white', outline = '')

master = Tk()
master.title( "Mnist Classifire" )
w = Canvas(master, 
           width=canvas_width, 
           height=canvas_height, bg = 'grey1', bd = 1)
w.pack(expand = YES, fill = BOTH)
w.bind( "<B1-Motion>", paint )

message = Label( master, text = "Press and Drag the mouse to draw" )
message.pack( side = BOTTOM )

w.update()

def keyDown(k):
	if k.char == 'c':
		w.delete("all")
	if k.char == 's':
		print("Saved successfully")
		#w.postscript(file="myNumber.ps", colormode='color')
		im = grab(bbox = (100, 200, canvas_width, canvas_height))
		im.show()
		#ImageGrab.grab((0, 0, w.width, w.height)).save(imageFilename + '.jpg')	

master.bind('<KeyPress>', keyDown)
    
mainloop()'''
