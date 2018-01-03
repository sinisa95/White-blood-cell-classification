from  data import loadImages
from neural_network import fitModel
from neural_network import predict
from neural_network import loadModel

epochs = 40
batch_size = 25

def main():


    model = None
    while True:
        try:
            print("\n1.Process images ")
            print("2.Fit new model")
            print("3.Load existing model")
            print("4.Predict test data")
            select = int(input("Please enter a number: "))

            if select == 1:
                print("\n\tProccesing train images...")
                loadImages("images/TRAIN", True)
                print("\tDONE! Train images are processed.")

                print("\n\tProccesing test images...")
                loadImages("images/TEST", True)
                print("\tDONE! Test images are loaded.")

            elif select == 2:
                model = fitModel(epochs, batch_size)
            elif select == 3:
                try:
                    model = loadModel("network.h5")
                except OSError:
                    print("\n!!! Model doesn't exist. You need first to fit model.")
            elif select == 4:
                if model is not None:
                    predict(model)
                else:
                    print("\n!!! You must first fit or load model")
        except ValueError:
            print("\n!!! That was no valid number.  Try again...")

main()









