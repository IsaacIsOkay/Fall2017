import os

x=0
y=988
while (x<=924):
	if(os.path.isfile("/home/default/.keras/datasets/weedspic/lettuce/image/%04d.bmp" % (x)) and
   	   os.path.isfile("/home/default/.keras/datasets/weedspic/lettuce/label/%04d.bmp" % (x))):
		old_file = os.path.join("/home/default/.keras/datasets/weedspic/lettuce/image", "%04d.bmp" % (x))
		new_file = os.path.join("/home/default/.keras/datasets/weedspic/lettuce/image_final", "%04d.bmp" % (y))
		os.rename(old_file, new_file)
	
		old_file = os.path.join("/home/default/.keras/datasets/weedspic/lettuce/label", "%04d.bmp" % (x))
		new_file = os.path.join("/home/default/.keras/datasets/weedspic/lettuce/label_final", "%04d.bmp" % (y))
		os.rename(old_file, new_file)
                y+=1
	x += 1
