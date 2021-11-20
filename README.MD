#  **Face mask detector**


I made this program to detect whether someone is wearing a facemask through a live video stream. I built my model on the MobileNetV2 model wich shows great improvemnts in accuracy over the custom solution in `version 1`.i  trained the model with TensorFlow on a public dataset provided by Kaggle . Opencv is used to create and feed each frame ino the original model.

To find the notebook used to train the model, visit: https://www.kaggle.com/reeganviljoen/face-mask-dtection 

## **Instalation**

### **Linux**


- **Debain based distros(eg Ubuntu)**
    1. Install python: `sudo apt install python3`/`sudo apt-get install python3`

    2. install pip(python package manager): `sudo apt install python3-pip`/`sudo apt install python3-pip`

    3. install opencv 2 : `pip3 install opencv-python`

    4. Install tensorflow : `pip3 install tensorflow`

    5. Install imutils: ` pip3 install imutils`

    6. Downlaod the model from my google drive(its to big for github): https://drive.google.com/drive/folders/1VLYJh1CotZViI7uQ1KPlZH-LeTed45O4?usp=sharing
### **Windows**
 
 1. Install python: Follow this link and the instructions that follow: https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe


2. install opencv 2 : `pip3 install opencv-python`

3. Install tensorflow : `pip3 install tensorflow`

4. Install imutils: ` pip3 install imutils`

 6. Downlaod the model from my google drive(its to big for github): https://drive.google.com/drive/folders/1VLYJh1CotZViI7uQ1KPlZH-LeTed45O4?usp=sharing

## **Running the program**

Before running your program it is important to note that opencv indexes your camera id from 0 ,but in this second version if have included a utility to help you find the right cmera number
so now you can test away with `camtest.py`

**Run the program:** `python3 main.py`

 after running the program follow the prompts

## **Contributing**

Contributions to this project are welcomed and encouraged!

**Important!:** All contributors are xepcted to follow the code of conduct listed in the code_of_conduct.md file to prevent abuse when contributing to this project

 

