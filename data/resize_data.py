import cv2
import os

resized_path = "./Twdne-128/"

def load_images_path_from_folder(folder):
    path = []
    i = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.resize(img, dsize=(128,128))
            img_number = f"{i:06}";
            print(img_number)
            cv2.imwrite(resized_path+ img_number +".jpg", img)
            i += 1

    return

if __name__ == '__main__':
    load_images_path_from_folder("./twdne")
    print("Finished")