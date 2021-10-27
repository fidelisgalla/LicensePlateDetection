
In this repo, we apply a method to firstly detect the location of license plate and then recognize the character inside it. 
1. For detection of license plate, we use YOLO. 
2. After detection, we apply some image processing
3. We use some image processing technique for conducting image processing
4. The next step is to conduct four point transformation on the image. The purpose of four point transform is to ease the recognition of an image. For the purpose of four point transformation, we use CRAFT algoritm to identify the corner of image.
5. The last step is character recognition, we use PyTesseract algoritm for doing that.

This repo is still in development for better result
