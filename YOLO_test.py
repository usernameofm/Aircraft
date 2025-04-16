from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO('.\\runs\detect\\train37\weights\\best.pt')

    model.predict('Your_True_Path', save=True)



