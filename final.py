import argparse

def final(parents):
    print("final method")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=str, required=True)
    args = parser.parse_args()
    final(args.parents)
    