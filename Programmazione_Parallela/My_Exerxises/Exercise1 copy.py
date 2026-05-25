def f(x) :
	while True :
		yield lambda y : x + y
		x += 1
h = f(0)
for i in range (5) :
	g = next (h)
	print (g(i))