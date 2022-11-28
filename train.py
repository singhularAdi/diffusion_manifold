from models import LeNet, LeNetPL


def main():
    lenet = LeNet()
    model = LeNetPL(lenet)

if __name__=='__main__':
    main()
