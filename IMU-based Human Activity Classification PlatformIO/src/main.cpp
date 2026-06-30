#include "board_setup.h"

// Core TensorFlow Lite Micro Headers (Using modern local inclusions)
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"       // Modern replacement for micro_error_reporter.h
#include "tensorflow/lite/schema/schema_generated.h" // Desktop-aligned verified subpath

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

// Global TFLite objects (Statically allocated to prevent heap allocation/fragmentation)
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroMutableOpResolver<6> op_resolver; // Size optimized to exactly what your model needs
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Memory allocation buffer arena for network tensor processing layers
const int kTensorArenaSize = 25 * 1024; // 25 KB
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize]; // Aligned to 16 bytes for hardware vector speedup

// Scaling parameters (mean and standard deviation) from StandardScaler
const float g_mean[] = {1906.18630431, 5231.88101198, -4316.21812981, 10.75862685, -274.85723225, -202.32853567};
const float g_std[] = {6616.99734886, 12900.50103468, 5838.89261839, 4586.6168787, 6332.61625367, 6228.89244874};

// Placeholder for IMU data acquisition.
void read_imu_data(float* input_data) {
    // Populate raw data vector mapping components: [ax, ay, az, gx, gy, gz]
    input_data[0] = -1248; // ax
    input_data[1] = 14736; // ay
    input_data[2] = -6780; // az
    input_data[3] = 1057;  // gx
    input_data[4] = 1422;  // gy
    input_data[5] = 2038;  // gz
}

void setup() {
    tflite::InitializeTarget();
    
    // UART terminal communication initializer setup 
    // Note: Serial interface functions are standard Arduino framework mappings. 
    // In plain stm32cube workloads, replace with your custom UART_printf routine if needed.
    Serial.begin(115200); 
    while (!Serial); 

    MicroPrintf("Initializing TFLite Micro...");

    // Map the model into a usable data structure
    model = tflite::GetModel(enhanced_activity_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch! Expected %d, got %d.",
                    TFLITE_SCHEMA_VERSION, model->version());
        return;
    }

    // Register only the specific mathematical nodes required by your ML architecture
    op_resolver.AddFullyConnected(); 
    op_resolver.AddSoftmax();

    // Build static persistent instance of interpreter handler context engine
    static tflite::MicroInterpreter static_interpreter(
        model, op_resolver, g_tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    // Allocate tensors from the memory arena
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed!");
        return;
    }

    // Fetch operational array structures
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    MicroPrintf("TFLite Micro initialized successfully!");
}

void loop() {
    delay(2500); // 2.5 seconds loop interval

    MicroPrintf("\nReading IMU data...");
    float raw_imu_data[6];
    read_imu_data(raw_imu_data);

    // Normalize incoming input features through the standard scaling parameters
    for (int i = 0; i < 6; ++i) {
        input_tensor->data.f[i] = (raw_imu_data[i] - g_mean[i]) / g_std[i];
    }

    // Run execution pass over internal layers
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed!");
        return;
    }

    // Capture tensor score layout metrics
    float* output_scores = output_tensor->data.f;

    // Extract optimal index classification mapping
    float max_score = -1.0f;
    int predicted_activity_index = -1;
    for (int i = 0; i < 6; ++i) {
        if (output_scores[i] > max_score) {
            max_score = output_scores[i];
            predicted_activity_index = i;
        }
    }

    // Broadcast results out to physical serial interface
    Serial.print("Predicted Activity: ");
    if (predicted_activity_index != -1) {
        Serial.print(activity_labels_cpp[predicted_activity_index]);
        Serial.print(" (Score: ");
        Serial.print(max_score, 4); 
        Serial.println(")");
    } else {
        Serial.println("Could not determine activity.");
    }
}