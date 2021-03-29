import os

dirs = os.listdir(".")
dirs = [name for name in dirs if (name[0]=='0' and len(name)==8)]

for kkk in dirs:
    dirs2 = os.listdir(kkk)
    print(kkk)
    print(len(dirs2))

    count1 = 0
    count2 = 0
    count3 = 0
    for i in dirs2:
        n1 = kkk + "/" + i +"/model.obj"
        n2 = kkk + "/" + i +"/model.binvox"
        n3 = kkk + "/" + i +"/model_depth_fusion.binvox"

        if os.path.exists(n1):
            count1 += 1
        if os.path.exists(n2):
            count2 += 1
        if os.path.exists(n3):
            count3 += 1
    
    print(count1)
    print(count2)
    print(count3)
    print()

        