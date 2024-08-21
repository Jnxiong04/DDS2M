import pickle
import sys
import os

def main():
    denoise_type = 'degrade'
    path = os.path.join('results/{}'.format(denoise_type), 'all_stats')
    print("getting results for " + denoise_type)
    with open(path, 'rb') as f:
        results = pickle.load(f)
    psnrs = []
    for res in results:
        psnrs.append(res['psnr_best'])
    print(len(psnrs))
    print(psnrs)
    return 0

if __name__ == "__main__":
    sys.exit(main())
