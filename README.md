This project is a Human Activity Recognition (HAR) system via a Machine Learning Model using Tensorflow Lite. 
I will explain methodologies soon. 

The hardware used is 
      1. STM32L476RG microcontroller
      2. BMI270 sensor from sparkfun

The sofware used is 
      1. VSCode
      2. PlatformIO
      3. Google Colab
      4. Github

STEP ONE:
Generate your machine learning model using the google colab ipynb file. Pull from any data source you would like. Initially, I used the IMU-Based Human Activity Recognition Dataset (https://data.mendeley.com/datasets/fcnmmsn857/3). I am going to gather my own data soon in the following categories (sitting, walking, and running). I like these three categories because it should be easy for my model to classify. I am currently working on collecting my data and will report how I do so. Additionally, the Tensorflow Lite library is not a readily available library in VSCode so I leveraged Google Colab so you do not have to download the entire library locally. I used a 3 layer model (32 -> 16 -> 6 neurons) at 100 epochs). I am working on adding cross validation. I included a confusion matrix and model testing as well pre-build. The rest saves the model as a header file for the next steps. 

STEP TWO:
Use VSCode and Platform.IO to connect to the STM32L476RG and BMI270 IMU Sensor. Using the header files and cpp files, you must build (takes up to 15min from a clean project) and then upload to the STM32L476RG. You must press the reset button and the first 10 lines will let you know if the sensor and board get configured properly. Currently, the code polls every 2.5s and outputs the predicted activity being done in the serial monitor. 
