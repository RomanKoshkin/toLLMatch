import sys

dict = {}
for line in sys.stdin:
    line = line.strip().replace('\n','').split('\t')
    dict[int(line[0])]=line[1]
    
sorted_list = sorted(dict.items(),key=lambda x:x[0])

for item in sorted_list:
    print(item[1])
