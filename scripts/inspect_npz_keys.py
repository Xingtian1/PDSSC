# -*- coding: utf-8 -*-
import sys
import numpy as np


def main():
    for path in sys.argv[1:]:
        data = np.load(path, allow_pickle=True)
        print(path)
        print(list(data.files))
        for key in data.files:
            arr = data[key]
            print(" ", key, getattr(arr, "shape", None), getattr(arr, "dtype", type(arr)))
        print("-" * 60)


if __name__ == "__main__":
    main()
