import LoadData
import NN_model
import training
import testing

def main():
    train_data, test_data = LoadData.loadData()
    train_data = train_data

    model = NN_model.MNIST_Model()
    model = model.to('cuda')

    training.training(model, train_data)
    testing.testing(model, test_data)


if __name__ == "__main__":
    main()



