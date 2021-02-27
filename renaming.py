import os

def main():
    i = 0
    path = "/Users/brownies/Desktop/BA/BIG_Project/Videos/DB/Test/Fight/"
    for filename in os.listdir(path):
        my_dest = "TE_Fight_" + str(i) + ".avi"
        my_source = path + filename
        my_dest = path + my_dest
        os.rename(my_source, my_dest)
        i += 1


if __name__ == '__main__':
    main()
