import sys  
print(sys.path)
paths = sys.path
for path in sys.path:
    sys.path.remove(path)