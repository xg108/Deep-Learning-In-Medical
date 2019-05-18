##python批量更换后缀名

import os


def convertjpg():
    # rootdir = 'data/VOCdevkit/VOC2007/JPEGImages/'
    rootdir = 'data/add_shuju/'
    list = os.listdir(rootdir)
    for filename in list:
        portion = filename.split('.')
        oldpath = rootdir + filename
        if (portion[1] == 'JPG'):
            portion[1] = 'jpg'
            name = portion[0] + '.' + portion[1]
            newpath = rootdir + name
            os.rename(oldpath, newpath)


def convertname():
    rootdir = 'data/normaloral/'
    list = os.listdir(rootdir)
    length = len(list)
    for i in range(0, length):
        portion = list[i].split('.')
        oldpath = rootdir + list[i]
        newname = str(i) + '.' + portion[1]
        newpath = rootdir + newname
        os.rename(oldpath, newpath)


if __name__ == '__main__':
    convertjpg()
