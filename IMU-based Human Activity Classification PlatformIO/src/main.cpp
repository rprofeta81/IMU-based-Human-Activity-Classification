#include "board_setup.h"

#define DO_FOREVER          while(1)
#define POLLING_DELAY_MS    2500  // Poll every 2.5 seconds

// Adjust tensor arena size if your new model is larger than the old sine-wave model
#define TENSOR_ARENA_SIZE   8 * 1024 

// Adjust these to match your model's exact expected dimensions
#define IMU_INPUT_ELEMENTS  6     // For example: Accel X, Y, Z + Gyro X, Y, Z
#define NUM_CLASSES         4     // For example: 0=Rest, 1=Walk, 2=Run, 3=Stairs

// Standard TensorFlow Lite Micro headers (Uncomment if needed)
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Your custom activity model header
#include "enhanced_activity_model.h"

// Helper function to find the index with the highest probability
int get_argmax(float* array, int size) {
    int max_idx = 0;
    float max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// Map the argmax index to a human-readable string
const char* get_activity_name(int class_index) {
    switch(class_index) {
        case 0: return "Resting";
        case 1: return "Walking";
        case 2: return "Running";
        default: return "Unknown Activity";
    }
}

int main(void) {
    HAL_Init();
    init_GPIO_pins();
    init_UART2();
    init_TIM2();
    // init_IMU(); // Ensure you initialize your specific IMU driver here (SPI/I2C)

    const tflite::Model* model = tflite::GetModel(enhanced_activity_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        UART_printf("Model version mismatch! Expected %d, got %d\n", TFLITE_SCHEMA_VERSION, model->version());
        return -1;
    }
    
    // Note: If your model uses operations other than FullyConnected and Relu (like Conv1D),
    // you will need to add those operators here or use AllOpsResolver.
    tflite::MicroMutableOpResolver<3> resolver;
    if (resolver.AddFullyConnected() != kTfLiteOk || 
        resolver.AddRelu() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk) { // Softmax is common for final classification layers
        UART_printf("Failed to add all the ops.\n");
        return -1;
    }

    // Keep aligned to 16 bytes for CMSIS-NN/ARM optimization
    alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];

    tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    tflite::MicroInterpreter* interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        UART_printf("Failed to allocate tensors.\n");
        return -1;
    }

    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);

    UART_printf("IMU Activity Classifier Initialized Successfully!\n");

    DO_FOREVER {
        uint32_t start_ms = HAL_GetTick();

        // 1. Read your physical IMU data into variables here
        float ax = 0.0f, ay = 0.0f, az = 0.0f; 
        // Example: IMU_Read_Accel(&ax, &ay, &az);

        // 2. Load the IMU data into the model input tensor
        // (Modify this indexing depending on if your model expects a time-series buffer)
        input->data.f[0] = ax;
        input->data.f[1] = ay;
        input->data.f[2] = az;
        // Fill the rest up to IMU_INPUT_ELEMENTS...

        // 3. Run Inference
        if (interpreter->Invoke() != kTfLiteOk) {
            UART_printf("Inference Failed!\n");
            HAL_Delay(POLLING_DELAY_MS);
            continue;
        }

        // 4. Process Classification Results
        int predicted_class = get_argmax(output->data.f, NUM_CLASSES);
        float confidence = output->data.f[predicted_class];
        uint32_t execution_time = HAL_GetTick() - start_ms;

        // 5. Print Results over Serial
        UART_printf("Predicted: %s (Confidence: %d%%) [Took %ums]\n", 
                    get_activity_name(predicted_class), 
                    (int)(confidence * 100), 
                    execution_time);

        // 6. Wait 2.5 seconds before polling again
        HAL_Delay(POLLING_DELAY_MS);
    }
}