import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="math")
parser.add_argument("--n", type=int, default=30)

if __name__ == "__main__":
    # 비어있는 텍스트파일을 n개 만큼 만드는 코드
    # 텍스트파일은 1.txt ... n.txt로 저장됨
    args = parser.parse_args()
    
    for i in range(1, args.n+1):
        path = os.path.join(args.dir, f"{i}.txt")
        with open(path, "w") as f:
            f.write("")
    
    