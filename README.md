# Gender-Classifier

# 1.What is it?
It's exactly what you think- The input is a picture with some people in it.
The output is bounding boxes around the people in it, and a gender classification, hopefully the correct one.

In fact, as far as I know this is currently (8-Sep-2017) the world's best Gender Classifier, improving significantly(cutting the error rate by 50%) the results obtained in this article http://www.openu.ac.il/home/hassner/projects/cnn_agegender/
exceeding 93%.
Having said that, there's still a lot of room for improvement!

#2.How do I run it?

After cloning the project, and meeting the requirements, all you need to do is put images at the test folder, and then simply 
python3 classify.py
This will literally show you the pictures with bounding boxes and classification


Requirements:
  - PyTorch
  - TorchVision
  - OpenCV
  - dlib
  - imutils (the face alignment algo is based on http://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)
