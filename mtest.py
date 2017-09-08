from imutils.face_utils import FaceAligner,rect_to_bb
from imutils import face_utils
# import argparse
import imutils
import dlib
from imutils.face_utils import FaceAligner
import os
import copy



GET_PLURAL_FACE=True
IMAGE_FILE_EXTENSION=".jpg"
# ALIGNED_PRE="aligned_"
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--directory", required=True,
#       help="path to directory of input images")
# args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)


IMAGES_DIRECTORY='test'


# load the input image, resize it, and convert it to grayscale

files=os.listdir(IMAGES_DIRECTORY)



def run_net_on_image(net,image,transform):
        transformed_image=transform(image)
        the_image=Variable(transformed_image).cuda().unsqueeze(0)
        net.eval()
        probabilities=net.forward(the_image)
        is_a_boy=probabilities[0][0].data<=probabilities[0][1].data
        is_a_boy=is_a_boy.cpu().numpy()[0]
        if is_a_boy==1:
                return 'Man'
        else:
                return 'Woman'

aligned_faces=[]

index=0
for filename in files:

        if not (len(filename)>len(IMAGE_FILE_EXTENSION) and filename[-len(IMAGE_FILE_EXTENSION):]==IMAGE_FILE_EXTENSION):
                continue
        index+=1
        # print('aligning picture number {}'.format(str(index)))
        fullFileName=os.path.join(IMAGES_DIRECTORY,filename)
        image = cv2.imread(fullFileName)
        image = imutils.resize(image, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_image=copy.deepcopy(image)


        # show the original input image and detect faces in the grayscale
        # image
        #cv2.imshow("Input", image)
        rects = detector(gray, 2)
        if len(rects)!=1:
                if len(rects)==0:
                        print('No face found!')
                        continue
                elif len(rects)>1:
                        print('More than one face found!')
                        if GET_PLURAL_FACE==False:
                                continue
                #continue

        # loop over the face detections
        for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                (x, y, w, h) = rect_to_bb(rect)
                if x<=0:
                        w=w+x
                        x=0
                if y<=0:
                        h=h+y
                        y=0
                if w<=0 or h<=0:
                        print('Bad Bounding Box!')
                        continue

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                faceAligned = fa.align(original_image, gray, rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # show the face number
                # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                #       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                
                for (x, y) in shape:
                        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)




                faceOrig = image #imutils.resize(image[y:y + h, x:x + w], width=256)
                

                #import uuid
                #f = str(uuid.uuid4())
                #cv2.imwrite("foo/" + f + ".png", faceAligned)

                # display the output images
                #cv2.imshow("Original", faceOrig)
                #cv2.imshow("Aligned", faceAligned)
                aligned_faces.append(faceAligned)
                # newFileFullName=os.path.join(args["directory"],ALIGNED_PRE+filename)
                # cv2.imwrite(newFileFullName,faceAligned)
                # os.remove(fullFileName)
                #cv2.waitKey(0)
                gender_found=run_net_on_image(net,faceAligned,fb_transform)
                cv2.putText(image,gender_found,(int(x+w/2),int(y-10)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                print(gender_found+'\n')
                cv2.imshow("ha",image)
                cv2.waitKey(0)
def cv2_imshow(img,transformed_tensor=False):
        if transformed_tensor==True:
                img=img.float()
                img = img / 2 + 0.5     # unnormalize
                npimg = img.numpy()
                npimg=np.transpose(npimg, (1, 2, 0))
                img=npimg*255
                #    cv2.imwrite('transformed.jpg',img)
