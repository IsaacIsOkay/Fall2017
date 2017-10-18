import os
from PIL import Image


train_path = os.path.expanduser('~/.keras/datasets/weedspic/lettuce/validation.txt')
train_path_img = os.path.expanduser('~/.keras/datasets/weedspic/lettuce/label_final/')
with open(train_path) as f:
	content = f.readlines()
content = [x.strip() for x in content]

for x in content:
	img = Image.open(train_path_img + '%s' % (x) + '.bmp')
	new_img = img
	new_img.save(os.path.expanduser('~/.keras/datasets/weedspic/lettuce/label/%s.png' % (x)), 'png')
print('converted')
