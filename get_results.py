import pickle
import sys
import os

def main():
    path = os.path.join('./results/test', 'x_demo_new.pickle')
    with open(path, 'rb') as f:
        results = pickle.load(f)
    psnrs = []
    for res in results:
        psnrs.append(res['psnr_best'])
    
    print(psnrs)
    return 0

if __name__ == "__main__":
    sys.exit(main())
