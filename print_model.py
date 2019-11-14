from model.model import create_model


def main():
    model = create_model()
    model.summary()


if __name__ == "__main__":
    main()
