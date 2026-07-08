```cpp
// BMI270.cpp
#include "BMI270.h"
#include "tensorflow/lite/micro/micro_log.h" // For MicroPrintf

// --- Use STM32 HAL delay function ---
#define DELAY_MS(ms) HAL_Delay(ms)

// --- Use TFLite Micro debug print function ---
#define DEBUG_PRINTF(...) MicroPrintf(__VA_ARGS__)

BMI270::BMI270(I2C_HandleTypeDef* hi2c) : _hi2c(hi2c),
                                         _device_address(BMI270_I2C_ADDR_PRIMARY),
                                         _accel_scale_factor(0.000244f * 9.80665f), /* Example for +/-16g range, convert to m/s^2 */
                                         _gyro_scale_factor(0.007629f) /* Example for +/-2000dps range */
{
    // Constructor initializes I2C handle and default scale factors
}

// --- Generic I2C Read/Write using STM32 HAL --- 
int8_t BMI270::i2c_read(uint8_t reg_addr, uint8_t *data, uint16_t len) {
    // HAL_I2C_Mem_Read(I2C_HandleTypeDef *hi2c, uint16_t DevAddress, uint16_t MemAddress, uint16_t MemAddSize, uint8_t *pData, uint16_t Size, uint32_t Timeout)
    if (HAL_I2C_Mem_Read(_hi2c, _device_address, reg_addr, I2C_MEMADD_SIZE_8BIT, data, len, HAL_MAX_DELAY) != HAL_OK) {
        DEBUG_PRINTF("I2C Read Error: Reg 0x%02X\r\n", reg_addr);
        return -1;
    }
    return 0;
}

int8_t BMI270::i2c_write(uint8_t reg_addr, uint8_t *data, uint16_t len) {
    // HAL_I2C_Mem_Write(I2C_HandleTypeDef *hi2c, uint16_t DevAddress, uint16_t MemAddress, uint16_t MemAddSize, uint8_t *pData, uint16_t Size, uint32_t Timeout)
    if (HAL_I2C_Mem_Write(_hi2c, _device_address, reg_addr, I2C_MEMADD_SIZE_8BIT, data, len, HAL_MAX_DELAY) != HAL_OK) {
        DEBUG_PRINTF("I2C Write Error: Reg 0x%02X\r\n", reg_addr);
        return -1;
    }
    return 0;
}

int8_t BMI270::bmi270_set_command(uint8_t command) {
    return i2c_write(BMI270_CMD_ADDR, &command, 1);
}

int8_t BMI270::init() {
    uint8_t chip_id;

    DEBUG_PRINTF("Initializing BMI270...\r\n");

    // 1. Read Chip ID to verify communication
    if (i2c_read(BMI270_CHIP_ID_ADDR, &chip_id, 1) != 0) {
        DEBUG_PRINTF("Failed to read BMI270 Chip ID\r\n");
        return -1;
    }

    if (chip_id != 0x24) { // BMI270/BMI160/BMX160 Chip ID is 0x24. Check datasheet.
        DEBUG_PRINTF("BMI270 Chip ID mismatch! Expected 0x24, got 0x%02X\r\n", chip_id);
        return -1;
    }
    DEBUG_PRINTF("BMI270 Chip ID: 0x%02X (Verified)\r\n", chip_id);

    // 2. Perform a software reset
    uint8_t cmd_soft_reset = 0xB6; // Command for soft reset
    if (bmi270_set_command(cmd_soft_reset) != 0) {
        DEBUG_PRINTF("Failed to soft reset BMI270\r\n");
        return -1;
    }
    DELAY_MS(150); // Wait for boot-up time as per datasheet

    // 3. Power on accelerometer and gyroscope
    // Set ACC_PWR_CONF to Normal mode (0x00)
    // Set GYR_PWR_CONF to Normal mode (0x00)
    // This is often done by writing to the PWR_CONF register (0x7C).
    // For BMI270, ensure that power modes for accel and gyro are set correctly.
    // Typically, 0x00 puts both in normal mode.
    uint8_t pwr_conf_data = 0x00; 
    if (i2c_write(BMI270_PWR_CONF_ADDR, &pwr_conf_data, 1) != 0) {
        DEBUG_PRINTF("Failed to set power configuration\r\n");
        return -1;
    }
    DELAY_MS(50); // Wait for power-up to stabilize

    // 4. Configure Accelerometer (e.g., ODR=100Hz, Range=+/-16g, BW=ODR/2)
    // Refer to BMI270 datasheet for specific register values for your desired configuration
    // Example: 0x28 for ODR_ACCEL=100Hz, 0x04 for G_RANGE=16g (ACC_RANGE, check actual register bits)
    uint8_t acc_conf_data[] = {0x28, 0x04}; 
    if (i2c_write(BMI270_ACC_CONF_ADDR, acc_conf_data, 2) != 0) {
        DEBUG_PRINTF("Failed to set accelerometer configuration\r\n");
        return -1;
    }
    // Update scale factor if range changes, e.g., for +/-16g (this example matches 16g)
    _accel_scale_factor = 0.000244f * 9.80665f; 

    // 5. Configure Gyroscope (e.g., ODR=100Hz, Range=+/-2000dps, BW=ODR/2)
    // Refer to BMI270 datasheet for specific register values
    // Example: 0x28 for ODR_GYRO=100Hz, 0x00 for DPS_RANGE=2000dps (GYR_RANGE, check actual register bits)
    uint8_t gyr_conf_data[] = {0x28, 0x00}; 
    if (i2c_write(BMI270_GYR_CONF_ADDR, gyr_conf_data, 2) != 0) {
        DEBUG_PRINTF("Failed to set gyroscope configuration\r\n");
        return -1;
    }
    // Update scale factor if range changes, e.g., for +/-2000dps (this example matches 2000dps)
    _gyro_scale_factor = 0.007629f; 

    DEBUG_PRINTF("BMI270 Initialized Successfully!\r\n");
    return 0;
}

int8_t BMI270::readAccelerometer(float* x, float* y, float* z) {
    uint8_t raw_data[6]; // X_LSB, X_MSB, Y_LSB, Y_MSB, Z_LSB, Z_MSB
    // Read 6 bytes starting from ACC_DATA_X_LSB_ADDR
    if (i2c_read(BMI270_ACC_DATA_X_LSB_ADDR, raw_data, 6) != 0) {
        DEBUG_PRINTF("Failed to read accelerometer data\r\n");
        return -1;
    }

    // Combine LSB and MSB to form 16-bit signed integers
    int16_t raw_acc_x = (int16_t)(raw_data[1] << 8 | raw_data[0]);
    int16_t raw_acc_y = (int16_t)(raw_data[3] << 8 | raw_data[2]);
    int16_t raw_acc_z = (int16_t)(raw_data[5] << 8 | raw_data[4]);

    // Apply scale factor to convert raw data to physical units
    *x = (float)raw_acc_x * _accel_scale_factor;
    *y = (float)raw_acc_y * _accel_scale_factor;
    *z = (float)raw_acc_z * _accel_scale_factor;

    return 0;
}

int8_t BMI270::readGyroscope(float* x, float* y, float* z) {
    uint8_t raw_data[6]; // X_LSB, X_MSB, Y_LSB, Y_MSB, Z_LSB, Z_MSB
    // Read 6 bytes starting from GYR_DATA_X_LSB_ADDR (0x12 for BMI270)
    if (i2c_read(BMI270_GYR_DATA_X_LSB_ADDR, raw_data, 6) != 0) { 
        DEBUG_PRINTF("Failed to read gyroscope data\r\n");
        return -1;
    }

    // Combine LSB and MSB to form 16-bit signed integers
    int16_t raw_gyr_x = (int16_t)(raw_data[1] << 8 | raw_data[0]);
    int16_t raw_gyr_y = (int16_t)(raw_data[3] << 8 | raw_data[2]);
    int16_t raw_gyr_z = (int16_t)(raw_data[5] << 8 | raw_data[4]);

    // Apply scale factor to convert raw data to physical units
    *x = (float)raw_gyr_x * _gyro_scale_factor;
    *y = (float)raw_gyr_y * _gyro_scale_factor;
    *z = (float)raw_gyr_z * _gyro_scale_factor;

    return 0;
}

int8_t BMI270::readSensorData(bmi270_sensor_data_t* data) {
    if (readAccelerometer(&data->acc_x, &data->acc_y, &data->acc_z) != 0) return -1;
    if (readGyroscope(&data->gyr_x, &data->gyr_y, &data->gyr_z) != 0) return -1;
    return 0;
}
```