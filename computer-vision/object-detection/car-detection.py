### install opencv version 3.4.2 for this exercise, 
### if you have a different version of OpenCV please switch to the 3.4.2 version
# !{sys.executable} -m pip install opencv-python==3.4.2.16
import urllib.request
import cv2
#print(cv2.__version__)
from matplotlib import pyplot as plt

# Image display function
def plt_show(image, title="", gray = False, size = (12,10)):
    from pylab import rcParams
    temp = image 
    
    #convert to grayscale images
    if gray == False:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    
    #change image size
    rcParams['figure.figsize'] = [10,10]
    #remove axes ticks
    plt.axis("off")
    plt.title(title)
    plt.imshow(temp, cmap='gray')
    plt.show()
#
# Function to detect cars in an image
def detect_obj(image):
    #clean your image
    plt_show(image)
    ## detect the car in the image
    print("Detecting cars...")
    object_list = detector.detectMultiScale(image)
    print("Number of cars found: ", len(object_list))
    print(object_list)
    #for each car, draw a rectangle around it
    for obj in object_list: 
        (x, y, w, h) = obj
        cv2.rectangle(image, (x, y), (x + w, y + h),(255, 0, 0), 2) #line thickness
    ## lets view the image
    plt_show(image)
#
# Load a pretrained classifier 
## read the url
#haarcascade_url = 'https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml'
haar_name = "cars.xml"
#urllib.request.urlretrieve(haarcascade_url, haar_name)
#
detector = cv2.CascadeClassifier(haar_name) #load the classifier
## we will read in a sample image
#image_url = "https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/CV0101/Dataset/car-road-behind.jpg"
image_name = "car-road-behind.jpg"
#urllib.request.urlretrieve(image_url, image_name)
image = cv2.imread(image_name)
#
# plt_show(image, "Original Image")
#
# detect cars in the image
detect_obj(image)

# Work on figuring out how to get the car detection to work with the image