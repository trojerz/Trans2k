from glob import glob
import shutil
c = glob("*/", recursive = True)

write_files = list()

for file_ in c:
    try:
        with open(file_ + 'absence.label', "r") as g:
            if len(g.read()) == 0:
                shutil.rmtree(file_)
                print(f"file {file_} deleted")
            else:
                write_files.append(file_[:-1])
    except:
        pass

write_files = list(sorted(write_files))

with open("got10k_train_full_split.txt", "w") as t:
    for item in write_files:
        t.write("%s\n" % item)

with open('list.txt', "w") as g:
    for item in write_files:
        g.write("%s\n" % item)


