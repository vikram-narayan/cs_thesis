import pdb
f=open('temp.data','r')
beginning=[]
middle=[]
end=[]
i=0
for line in f:
    line=line[:len(line)-1]
    line=line.split(') (')
    if len(line)==1:
        line=line[:len(line)-1]
    for index,phrase in enumerate(line):
        if index==0:
            beginning.append(phrase)
        elif index==len(line)-1:
            end.append(phrase)
        else:
            middle.append(phrase)
    # print  len(beginning[i]) + len(middle[i]) + len(end[i]) == len(line)
f.close()

b=open('beginning.data','w')
for p in beginning:
    b.write(p+')')
    b.write('\n')
b.close()

b=open('middle.data','w')
for p in middle:
    b.write('('+p+')')
    b.write('\n')
b.close()

b=open('end.data','w')
for p in end:
    b.write('('+p)
    b.write('\n')
b.close()
