from argparse import ArgumentParser



def main(args):
 pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    main(args)