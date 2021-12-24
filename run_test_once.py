from BHIC import *

if __name__ == '__main__':
    # init the two stage model
    model = BHIC()
    # test image
    test_image_path = './image/1-1.jpg'
    # detect
    result = model.detect_once(test_image_path)
    print("result:", result)
