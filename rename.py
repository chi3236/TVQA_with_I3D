import os

path = './tvqa_subtitles'
for file in os.listdir(path):
    if file.startswith("s"):
        os.rename(path+'/'+file, path+'/'+"bbt_" + file)