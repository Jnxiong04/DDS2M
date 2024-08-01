import pickle
import sys
import os

def main():
    path = os.path.join('./results/test', 'x_demo.pickle')
    with open(path, 'rb') as f:
        results = pickle.load(f)
    psnrs = []
    for res in result:
        psnrs.append(res[psnr])
    
    print(psnrs)
    return 0

if __name__ == "__main__":
    sys.exit(main())
