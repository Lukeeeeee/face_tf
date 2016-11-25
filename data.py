import numpy
from PIL import Image
import cPickle

'''
img = Image.open('olivettifaces.gif')
img_ndarray = numpy.asarray(img, dtype='float64')/256
olivettifaces=numpy.empty((400,2679))
for row in range(20):
	for column in range(20):
		olivettifaces[row*20+column]=numpy.ndarray.flatten(img_ndarray [row*57:(row+1)*57,column*47:(column+1)*47])

olivettifaces_label=numpy.empty(400)
for label in range(40):
	olivettifaces_label[label*10:label*10+10]=label
olivettifaces_label=olivettifaces_label.astype(numpy.int)


write_file=open('olivettifaces.pkl','wb')  
cPickle.dump(olivettifaces,write_file,-1)  
cPickle.dump(olivettifaces_label,write_file,-1)  
write_file.close() 
'''
read_file=open('olivettifaces.pkl','rb')    
faces=cPickle.load(read_file)    
label=cPickle.load(read_file)    
read_file.close()   

#output data

train_data=numpy.empty((320,2679))  
train_label=numpy.empty((320,40))
test_data=numpy.empty((80,2679))  
test_label=numpy.empty((80,40))  

#print test_label[0]
#print test_label[1]

for i in range(40):  
	train_data[i*8:i*8+8]=faces[i*10:i*10+8]
	test_data[i*2:i*2+2]=faces[i*10+8:i*10+10]

	for j in range(10):
		if(j<8):
			train_label[i*8+j][i] = 1;
		else:
			test_label[i*2+j-8][i] = 1;

def return_train_data():
	return train_data, train_label
def return_test_data():
	return test_data[0:80], test_label[0:80]
def return_valid_data():
	return valid_data, valid_label

def return_next_batch(batch_size, pos):
	start = pos*batch_size
	end = start + batch_size
	return train_data[start:end], train_label[start:end]
	




