#  **Face mask detector**


This program was desighned to detect wheter someone is whearing a face-mask through a live video stream.A custom lightwigh CNN trained with Tensorflow on a public dataset provided by kaggle is used to detect wheter each face detected by the cv2 face detection dnn is whearing a mask

##To find the notebook used to traiin the model, visit: https://www.kaggle.com/reeganviljoen/face-mask-dtection##

## **Instalation**

### **Linux**


- **Debain based distros(eg Ubuntu)**
    1. Install python: `sudo apt install python3`/`sudo apt-get install python3`

    2. install pip(python package manager): `sudo apt install python3-pip`/`sudo apt install python3-pip`

    3. install opencv 2 : `pip3 install opencv-python`

    4. Install tensorflow : `pip3 install tensorflow`

    5. Install imutils: ` pip3 install imutils`

    6. unzip the model:

        6.1. install unzip: `sudo apt install unzip`

        6.2. unzip the model: `unzip mask_model\face_mask_model.zip `
### **Windows**
 
 1. Install python: Follow this link and the instructions that follow: https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe


2. install opencv 2 : `pip3 install opencv-python`

3. Install tensorflow : `pip3 install tensorflow`

4. Install imutils: ` pip3 install imutils`

5. unzip `mask_model\face_mask_model.zip`

## **Running the program**

Before running your program it is important to note that opencv indexes your camera id from 0 , and it may take a few attempts to find the right camera in a mutiple camera set up

**Run the program:**`python3 main.py`
 after running the program follow the prompts

## **Contributing**

Contributions to this project are welcomed and encouraged!

**Important!:** All contributors are xepcted to follow the code of conduct listed in the code_of_conduct.md file to prevent abuse when contributing to this project

 



# Face-Mask-detector
