import os

from PIL import Image

from __init__ import DOG_JINMAO_DIR, DOG_SUMU_DIR


def get_aver_dog(Path: str):
    w, h, w_a, h_a = 0, 0, 0, 0
    for img in os.listdir(Path):
        img_pic = None
        im = Image.open(Path + "/" + img)
        w = w + im.size[0]
        h = h + im.size[1]
        w_a = 585
        h_a = 460
        # if im.size[0] <= w_a and im.size[1]<= h_a:
        # 	img_pic = im.crop((0,0,w_a,h_a))
        # if im.size[0] <= w_a and im.size[1] > h_a:
        # 	img_pic = im.crop((0,im.size[1]-h_a,w_a,im.size[1]))
        # if im.size[0] > w_a and im.size[1] <= h_a:
        # 	img_pic = im.crop((im.size[0]-w_a,0,im.size[0],h_a))
        # else:
        # 	img_pic = im.crop((im.size[0]-w_a,im.size[1]-h_a,w_a,h_a))
        img_pic = im.crop((0, 0, w_a, h_a))
        img_pic.save(Path + "/" + "ok_%s" % (img))
        print(img_pic)
    w = w // (len(os.listdir(Path)))
    h = h // (len(os.listdir(Path)))
    print(Path, w, h)


# get_aver_dog(DOG_SUMU_DIR)
# get_aver_dog(DOG_JINMAO_DIR)
