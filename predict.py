# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

labels=[6,9,5,4,6,5,6,6,7,6,7,5,5,6,3,8,8,4,4,6,7]

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# load an image and predict the class
def run_example():
    # load the image
    images=[]
    for i in range(4,25):
        img = load_image('images-cropped1/111x77/resized/test-crop-07-'+str(i).zfill(2)+'.jpg')
        images.append(img)
    print('Images are ready')
    # load model
    model = load_model('final_model.h5')
    print('Model loaded')
    # predict the class
    i = 4
    print('No.| Pred.| Lab.| Result')
    for img in images:
        digit = model.predict_classes(img)
        res = 'good' if digit[0] == labels[i-4] else 'false'
        # if digit[0] == labels[i-4]:
        #     res = 'good'
        # else:
        #     res = 'false'
        print(str(i).zfill(2), '--', digit[0], '--', labels[i-4], '--', res)
        i += 1


# entry point, run the example
run_example()
