from os import listdir

debug = True

MODEL_NAME = "test_1908.h5"
RESULT_SAVING_PATH = "./results/hdf5mdl"

BATCH_SIZE = 16

CAT_PIC_PATH = "./image/kagglecatsanddogs_3367a/PetImages/Cat"
CAT_PATH = "image/kagglecatsanddogs_3367a/PetImages/Cat"
VALIDATION_PATH = "image/kagglecatsanddogs_3367a/PetImages/CatValidation"

DOG_PIC_WITDH = 585
DOG_PIC_HEIGHT = 460
DOG_JINMAO_DIR = "./image/dog/jinmao"
DOG_SUMU_DIR = "./image/dog/sumu"
DOG_JINMAO_SAMPLE_DIR = "./image/dog/jinmao-samples"
DOG_SUMU_SAMPLE_DIR = "./image/dog/sumu-samples"
test_jinmao = "./image/temp/jinmao.jpg"
test_sumu = "./image/temp/sumu.jpg"
test_dir = "./image/temp"

test_6_MODEL = "./results/hdf5mdl"
NUM_CLASSES = 2
CLASSES = ["man", "dog"]

SEP = "\n===================================================================\n"
# SIZE = len(listdir(CAT_PATH))


def write_log(log: str, path="./tmp/log/", file="test.log"):
    f = open(path + file, "a", encoding="utf-8")
    f.write(SEP)
    f.write(log)
    f.write(SEP)
    f.close()


# def load_data():
# 	global SIZE
# 	data = np.empty((SIZE,358,413,3),dtype="float32")
# 	label = np.empty((SIZE,357,412,16),dtype="uint8")
# 	#data = np.empty((SIZE,358,413),dtype="float32")
# 	#lable = np.empty((20,))
# 	imgs = listdir(CAT_PATH)
# 	num = SIZE
# 	print("Theis are %s cats "%(str(num)))
# 	for i in range(num):
# 		img = Image.open(CAT_PATH+"/"+imgs[i])
# 		arr = np.asarray(img,dtype="float32")
# 		data[i,:,:] = arr
# 		if i < 20:
# 			label[i] = int(imgs[i].split('.')[0])
# 	return data,label


haar_frontalface_default_xml = "haarcascade_frontalface_default.xml"
haar_frontalface_default_xml = "haarcascade_frontalface_alt2.xml"
harr_cascade_eye_xml = "haarcascade_eye.xml"
haar_frontalface_default_xml = (
    "C:\\Users\\xiaohaidan\\dog-face-check\\opencv-3.4.4\\opencv\\sources\\data\\haarcascades\\%s"
    % (haar_frontalface_default_xml)
)
harr_cascade_eye_xml = (
    "C:\\Users\\xiaohaidan\\dog-face-check\\opencv-3.4.4\\opencv\\sources\\data\\haarcascades\\%s"
    % (harr_cascade_eye_xml)
)
man_test_jpg = "./image/temp/one.1.jpg"
# man_test_jpg = "./image/temp/many.0.png"
haar_cascade_dir = "C:\\Users\\xiaohaidan\\dog-face-check\\opencv-3.4.4\\opencv\\sources\\data\\haarcascades"
haar_cascade_dir = "./model/opencv"
you_jpg = "./image/temp/you.jpg"
