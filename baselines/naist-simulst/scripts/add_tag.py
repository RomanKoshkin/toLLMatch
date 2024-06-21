import csv
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
tag = sys.argv[3]

with open(input_file, "r", newline="") as input_f, open(output_file, "w") as output_f:
    tsv_reader = csv.reader(input_f, delimiter="\t")
    tsv_writer = csv.writer(output_f, delimiter="\t", lineterminator='\n')

    header_row = next(tsv_reader)
    tsv_writer.writerow(header_row)
    
    for row in tsv_reader:
        row[3] = tag + row[3]
        tsv_writer.writerow(row)
