/*
  IMU Classifier
  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.
  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.
  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense Rev2 board.
  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry
  This example code is in the public domain.
*/

#include "Arduino_BMI270_BMM150.h"
#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"

const float accelerationThreshold = 2.5; // threshold of significant in G's
const int  numSamples = 20;

int samplesRead = 0;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {
  "ngua",
  "trai",
  "phải",
  "sap" 
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  float aX, aY, aZ;
  int type = 1;
  // // wait for significant motion
  // while (samplesRead == numSamples) {
  //   if (IMU.accelerationAvailable()) {
  //     // read the acceleration data
  //     IMU.readAcceleration(aX, aY, aZ);

  //     // sum up the absolutes
  //     float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
  //     // check if it's above the threshold
  //     if (aSum >= accelerationThreshold) {
  //       // reset the sample read count
  //       samplesRead = 0;
  //       break;
  //     }
  //   }
  // }

  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  std::vector<float> data1;
  while (true) {
    delay(500);
    if (IMU.accelerationAvailable()) {
      
      IMU.readAcceleration(aX, aY, aZ);

      Serial.println("-----------aX, aY, aZ-----------");
      Serial.println(aX);
      Serial.println(aY);
      Serial.println(aZ);

      
      // normalize the IMU data between 0 to 1 and store in the model's
      // input tensor
      float aX_new, aY_new, aZ_new;
      aX_new = ((aX * 10 + 12.0) / 24.0) ;
      aY_new = ((aY * 10 + 12.0) / 24.0) ;
      aZ_new = ((aZ * 10 + 12.0) / 24.0) ;


      // tflInputTensor->data.f[samplesRead * 3 + 0] = aX_new;
      // tflInputTensor->data.f[samplesRead * 3 + 1] = aY_new;
      // tflInputTensor->data.f[samplesRead * 3 + 2] = aZ_new;

      data1.push_back(aX_new);
      data1.push_back(aY_new);
      data1.push_back(aZ_new);

      
      samplesRead++;

      Serial.println("----------------------");
      Serial.println(samplesRead);
      Serial.println(data1.size());



      if (data1.size() == numSamples *3 ) {

        delay(2000);
        for (int i = 0; i < data1.size(); ++i) {
            tflInputTensor->data.f[i] = data1[i];
        }

        if (data1.size() >= 30) {
            data1.erase(data1.begin(), data1.begin() + 30);
        } else {
            data1.clear(); // Nếu data không có đủ 30 phần tử, xóa hết các phần tử.
        }

        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          while (1);
          return;
        }
        
        // Loop through the output tensor values from the model
        
        if(type == 1){
          type = 2;
        }else{
          type = 1;
        }

        for (int i = 0; i < NUM_GESTURES; i++) {
          Serial.print(i);
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i]);
        }
        Serial.println();
        samplesRead = 0;
      }
    }
  }
}