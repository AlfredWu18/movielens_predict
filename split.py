from random import shuffle

data=[]
f = open("ratings.csv","r")
f_out = open("rating_s.csv","w")
for row in f:
	if row.startswith("userId"):
		continue
	data.append(str(row))

shuffle(data)

f_out = open("rating_train.csv","w")
f_out.write("userId,movieId,rating,timestamp"+'\n')
for d in data[:70000]:
	f_out.write(d)
	f_out.flush()


f_out = open("rating_test.csv","w")
f_out.write("userId,movieId,rating,timestamp"+'\n')
for d in data[70000:]:
	f_out.write(d)
	f_out.flush()