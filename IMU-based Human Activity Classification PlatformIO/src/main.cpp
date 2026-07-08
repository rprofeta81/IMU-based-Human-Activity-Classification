#include "board_setup.h"
#include "enhanced_activity_model.h"

// TensorFlow Lite Micro Headers
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"       
#include "tensorflow/lite/schema/schema_generated.h" 

// defining activity labels 
const char* activity_labels_cpp[] = {
    "Walking Upstairs",
    "Walking Downstairs",
    "Walking",
    "Sitting",
    "Standing",
    "Jogging"
};

// Pointers and objects used by TensorFlow Lite Micro to manage the model, 
// perform inference, resolve operations, and interact with input/output data.
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroMutableOpResolver<6> op_resolver; 
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Memory allocation buffer arena for network tensor processing layers
// This is a common practice in memory-constrained embedded systems.
const int kTensorArenaSize = 25 * 1024; // 25 KB
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize]; // Aligned to 16 bytes for hardware vector speedup

// Scaling parameters to normalize the raw IMU data before feeding it into the neural network model.
const float g_mean[] = {1906.18630431, 5231.88101198, -4316.21812981, 10.75862685, -274.85723225, -202.32853567};
const float g_std[] = {6616.99734886, 12900.50103468, 5838.89261839, 4586.6168787, 6332.61625367, 6228.89244874};

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
    // target specific initialization for TensorFlow Lite Micro
    tflite::InitializeTarget();
    
    // Initialize required pins
    init_GPIO_pins();
    init_UART2();
    init_TIM2();

    UART_printf("Initializing TFLite Micro...\r\n");

    // Checks if the model's schema version matches the TFLite Micro library's expected version.
    model = tflite::GetModel(enhanced_activity_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch! Expected %d, got %d.",
                    TFLITE_SCHEMA_VERSION, model->version());
        return;
    }

    // Registers only "Fully Connected" and "Softmax" layers that my model uses
    op_resolver.AddFullyConnected(); 
    op_resolver.AddSoftmax();

    // Creates an interpreter instance, linking the model, op resolver, and the tensor arena.
    static tflite::MicroInterpreter static_interpreter(
        model, op_resolver, g_tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    // Allocate memory for all the model's tensors from the g_tensor_arena buffer
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed!");
        return;
    }

    // Gets pointers to the model's input and output tensors, which will be used to feed data in and read predictions out.
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    UART_printf("TFLite Micro initialized successfully!\r\n");
}



void loop() {
    // STM32 hardware SysTick Delay for 2.5s
    HAL_Delay(2500); 

    UART_printf("\r\nReading IMU data...\r\n");
    float raw_imu_data[6];
    read_imu_data(raw_imu_data);

    // Normalize incoming input features through the standard scaling parameters
    for (int i = 0; i < 6; ++i) {
        input_tensor->data.f[i] = (raw_imu_data[i] - g_mean[i]) / g_std[i];
    }

    // model performs its inference on the provided input data
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed!");
        return;
    }

    // retrieves the prediction probabilities from the model's output tensor
    float* output_scores = output_tensor->data.f;

    // inds the index of the output_scores array that has the highest value
    float max_score = -1.0f;
    int predicted_activity_index = -1;
    for (int i = 0; i < 6; ++i) {
        if (output_scores[i] > max_score) {
            max_score = output_scores[i];
            predicted_activity_index = i;
        }
    }

    // Broadcast results out to physical serial interface using native layout
    if (predicted_activity_index != -1) {
        UART_printf("Predicted Activity: %s (Score: %d.%04d)\r\n", 
                    activity_labels_cpp[predicted_activity_index],
                    (int)max_score, 
                    (int)((max_score - (int)max_score) * 10000));
    } else {
        UART_printf("Could not determine activity.\r\n");
    }
}



int main(void) {
    // Resets all peripherals, Initializes the Flash interface and the Systick.
    HAL_Init();
    
    // Called once to initialize everything
    setup();
    while (1) {
        // infinite loop to continuously read IMU data, perform inference, and output results
        loop();
    }
}