txtfile = open("new_keras10.txt",'r')
for i,line in enumerate(txtfile):
    if i == 4:
        line = line.split()
        print(line[:15])
    
