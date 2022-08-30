os.system("cls")
    data=[]
    with open("prediksi_PM.csv") as csv_file:
        csv_reader=csv.reader(csv_file,delimiter=",")
        for row in csv_reader:
            data.append(row)
    
    labels=data.pop(0)
    print("-"*52)
    print(f"{labels[0]}\t\t{labels[1]}\t{labels[2]}")
    print("-"*52)
    for row in sorted(data,reverse=True):
        print(f"{row[0]}\t{row[1]}\t{row[2]}")
 
