
import shutil
import os
#moving the files
folder = /KerasWorkspace/Fall2017/train
downloadFolder = /Downloads/LifeClef

type = null
number = 0

def getImType(filename):
	imType = #get the imType from an xml file
	return imType

for f in downloadFolder:
	if(f is an xml)
		type = getImType(f)
	else
		if(os.path.exists(f'{folder}/{type}'))
			shutil.copyfile(f, f'{folder}/{type}')
		else
			os.makedirs(f'{folder}/{type}')
			shutil.copyfile(f, f'{folder}/{type}')
			
#number of classes created	
for fold in folder:
	number++

print number
		
	
'''
source = '/path/to/source/'
dest1 = '/path/to/dest_folder'

files = os.listdir(source)

for f in files:
	shutil.move(source+f, dest1)
	
newpath = r'C:\Program Files\arbitrary' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
'''
