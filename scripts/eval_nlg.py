from nlgeval import compute_metrics
import sys
with open("hypo.txt", "w",encoding='utf-8') as f, open("refe.txt", "w",encoding='utf-8') as f2:
    file = open(sys.argv[1], mode='r', encoding='UTF-8') 
    while True:
        text_line = file.readline()
        if text_line:
            if text_line[0] == 'H':
                    f.write(text_line.split('\t')[2])
            if text_line[0] == 'T':
                    f2.write(text_line.split('\t')[1])
        else:
            break
metrics_dict = compute_metrics(hypothesis='hypo.txt',
                               references=['refe.txt'])
print(metrics_dict)