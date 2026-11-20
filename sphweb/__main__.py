from simweb.app import run


def main():
    run("sphweb", "sphweb.server:app", 8766)


if __name__ == "__main__":
    main()
