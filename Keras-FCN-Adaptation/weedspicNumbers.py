f = open('train.txt', 'w')



for x in range(0, 1183):
	f.write("%04d\n" % (x))
f.close()

