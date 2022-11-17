#串行执行
import time

def func1():
    for i in range(10000000):
        i+1

def func2():
    for i in range(10000000):
        i+1

start = time.time()
func1()
func2()
stop = time.time()
print(stop - start)


#基于yield并发执行
import time
def func1():
    while True:
        yield

def func2():
    g=func1()
    for i in range(10000000):
        i+1
        next(g)

start=time.time()
func2()
stop=time.time()
print(stop-start)