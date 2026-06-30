#include "board_setup.h"

// Core TensorFlow Lite Micro Headers
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // Replaces all_ops_resolver.h
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"       // Replaces micro_error_reporter.h
#include "tensorflow/lite/schema/schema_generated.h"

#include "enhanced_activity_model.h"

// Define activity labels (must match the order used during training)
const char* activity_labels_cpp[] = {
    "Walking Upstairs",
    "Walking Downstairs",
    "Walking",
    "Sitting",
    "Standing",
    "Jogging"
};

// Global TFLite objects
tflite::MicroErrorReporter tflite_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroMutableOpResolver<10>* op_resolver = nullptr; // Increased resolver size
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Create an arena for the model's tensors
// The size might need to be adjusted based on your model's complexity
const int kTensorArenaSize = 25 * 1024; // 25 KB, adjust if needed
uint8_t g_tensor_arena[kTensorArenaSize];

// Scaling parameters (mean and standard deviation) from StandardScaler
// These values must be obtained from your Python training script
const float g_mean[] = {1906.18630431, 5231.88101198, -4316.21812981, 10.75862685, -274.85723225, -202.32853567};
const float g_std[] = {6616.99734886, 12900.50103468, 5838.89261839, 4586.6168787, 6332.61625367, 6228.89244874};

// Placeholder for IMU data acquisition. Replace with your sensor code.
// This function should read raw IMU data (ax, ay, az, gx, gy, gz)
// and populate the input_data array.
void read_imu_data(float* input_data) {
    // For demonstration, we'll use a fixed set of values. 
    // In a real application, you would read from your IMU sensor here.
    // Example raw data (replace with actual sensor readings)
    input_data[0] = -1248; // ax
    input_data[1] = 14736; // ay
    input_data[2] = -6780; // az
    input_data[3] = 1057;  // gx
    input_data[4] = 1422;  // gy
    input_data[5] = 2038;  // gz

    // You could also simulate different activities for testing:
    // static int counter = 0;
    // if (counter % 2 == 0) { // Simulate walking upstairs
    //     input_data[0] = -1000; input_data[1] = 15000; input_data[2] = -7000;
    //     input_data[3] = 1000; input_data[4] = 1500; input_data[5] = 2000;
    // } else { // Simulate sitting
    //     input_data[0] = 500; input_data[1] = 100; input_data[2] = 9500;
    //     input_data[3] = 50; input_data[4] = 10; input_data[5] = 20;
    // }
    // counter++;
}

void setup() {
    tflite::InitializeTarget();
    Serial.begin(115200); // Initialize serial communication
    while (!Serial); // Wait for serial port to connect. Needed for native USB

    Serial.println("Initializing TFLite Micro...");

    // Map the model into a usable data structure
    model = tflite::GetModel(enhanced_activity_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(&tflite_error_reporter, "Model schema version mismatch! Expected %d, got %d.",
                             TFLITE_SCHEMA_VERSION, model->version());
        return;
    }

    // Pull in the all ops. You can optimize this by only including the ops you need.
    op_resolver = new tflite::MicroMutableOpResolver<10>();
    op_resolver->AddDense();
    op_resolver->AddFullyConnected(); // Dense is often aliased to FullyConnected
    op_resolver->AddSoftmax();
    // Add any other ops your model uses

    // Build an interpreter to run the model
    interpreter = new tflite::MicroInterpreter(
        model, *op_resolver, g_tensor_arena, kTensorArenaSize, &tflite_error_reporter);

    // Allocate tensors from the memory arena
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(&tflite_error_reporter, "AllocateTensors() failed!");
        return;
    }

    // Get pointers to the input and output tensors
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.println("TFLite Micro initialized successfully!");
}

void loop() {
    // Delay for 2-3 seconds
    delay(2500); // 2500 milliseconds = 2.5 seconds

    Serial.println("\nReading IMU data...");
    float raw_imu_data[6];
    read_imu_data(raw_imu_data);

    // Scale the input data using the pre-calculated mean and std dev
    for (int i = 0; i < 6; ++i) {
        input_tensor->data.f[i] = (raw_imu_data[i] - g_mean[i]) / g_std[i];
    }

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(&tflite_error_reporter, "Invoke failed!");
        return;
    }

    // Get the output scores (probabilities)
    float* output_scores = output_tensor->data.f;

    // Find the activity with the highest score
    float max_score = -1.0;
    int predicted_activity_index = -1;
    for (int i = 0; i < 6; ++i) {
        if (output_scores[i] > max_score) {
            max_score = output_scores[i];
            predicted_activity_index = i;
        }
    }

    Serial.print("Predicted Activity: ");
    if (predicted_activity_index != -1) {
        Serial.print(activity_labels_cpp[predicted_activity_index]);
        Serial.print(" (Score: ");
        Serial.print(max_score, 4); // Print score with 4 decimal places
        Serial.println(")");
    } else {
        Serial.println("Could not determine activity.");
    }
}