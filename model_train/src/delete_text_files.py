import os
import sys

if __name__ == '__main__':

    this_path = os.path.dirname(os.path.abspath(__file__))
    
    for dirpath, dirnames, filenames in os.walk(sys.argv[1]):
        for f in [ff for ff in filenames if ff.endswith(".txt")]:
            print("deleting file: {}".format(f))
            fil = dirpath + "/" + f
            os.remove(fil)
