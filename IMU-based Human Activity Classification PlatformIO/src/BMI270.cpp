#include <stdint.h>
#include "BMI270.h"
#include "tensorflow/lite/micro/micro_log.h" // For MicroPrintf

// --- Use STM32 HAL delay function ---
#define DELAY_MS(ms) HAL_Delay(ms)

// --- Use TFLite Micro debug print function ---
#define DEBUG_PRINTF(...) MicroPrintf(__VA_ARGS__)

BMI270::BMI270(I2C_HandleTypeDef* hi2c) : _hi2c(hi2c),
                                         _device_address(BMI270_I2C_ADDR_PRIMARY),
                                         _accel_scale_factor(0.000061f * 9.80665f), 
                                         _gyro_scale_factor(0.061035f)
{
}

int8_t BMI270::i2c_read(uint8_t reg_addr, uint8_t *data, uint16_t len) {
    if (HAL_I2C_Mem_Read(_hi2c, _device_address, reg_addr, I2C_MEMADD_SIZE_8BIT, data, len, 100) != HAL_OK) {
        DEBUG_PRINTF("I2C Read Error: Reg 0x%02X\r\n", reg_addr);
        return -1;
    }
    return 0;
}

int8_t BMI270::i2c_write(uint8_t reg_addr, uint8_t *data, uint16_t len) {
    if (HAL_I2C_Mem_Write(_hi2c, _device_address, reg_addr, I2C_MEMADD_SIZE_8BIT, data, len, 100) != HAL_OK) {
        DEBUG_PRINTF("I2C Write Error: Reg 0x%02X\r\n", reg_addr);
        return -1;
    }
    return 0;
}

int8_t BMI270::bmi270_set_command(uint8_t command) {
    return i2c_write(BMI270_CMD_ADDR, &command, 1);
}

int8_t BMI270::init() {
    uint8_t chip_id = 0;
    DEBUG_PRINTF("Initializing BMI270...\r\n");

    if (i2c_read(BMI270_CHIP_ID_ADDR, &chip_id, 1) != 0) {
        DEBUG_PRINTF("Failed to read BMI270 Chip ID\r\n");
        return -1;
    }

    if (chip_id != 0x24) {
        DEBUG_PRINTF("BMI270 Chip ID mismatch! Expected 0x24, got 0x%02X\r\n", chip_id);
        return -1;
    }
    DEBUG_PRINTF("BMI270 Chip ID: 0x%02X (Verified)\r\n", chip_id);

    uint8_t cmd_soft_reset = 0xB6;
    if (bmi270_set_command(cmd_soft_reset) != 0) {
        DEBUG_PRINTF("Failed to soft reset BMI270\r\n");
        return -1;
    }
    DELAY_MS(20); 

    uint8_t pwr_conf_data = 0x02; 
    if (i2c_write(BMI270_PWR_CONF_ADDR, &pwr_conf_data, 1) != 0) {
        DEBUG_PRINTF("Failed to set power configuration\r\n");
        return -1;
    }
    DELAY_MS(50);

    uint8_t pwr_ctrl_data = 0x0E; 
    if (i2c_write(0x7D, &pwr_ctrl_data, 1) != 0) {
        DEBUG_PRINTF("Failed to enable Accel/Gyro Power Engines\r\n");
        return -1;
    }
    DELAY_MS(10);

    uint8_t acc_conf_data[] = {0xA8, 0x00}; 
    if (i2c_write(BMI270_ACC_CONF_ADDR, acc_conf_data, 2) != 0) {
        DEBUG_PRINTF("Failed to set accelerometer configuration\r\n");
        return -1;
    }

    uint8_t gyr_conf_data[] = {0xA8, 0x00}; 
    if (i2c_write(BMI270_GYR_CONF_ADDR, gyr_conf_data, 2) != 0) {
        DEBUG_PRINTF("Failed to set gyroscope configuration\r\n");
        return -1;
    }

    DEBUG_PRINTF("BMI270 Initialized Successfully!\r\n");
    return 0;
}

int8_t BMI270::readAccelerometer(float* x, float* y, float* z) {
    uint8_t raw_data[6]; 
    if (i2c_read(BMI270_ACC_DATA_X_LSB_ADDR, raw_data, 6) != 0) return -1;

    int16_t raw_acc_x = (int16_t)(raw_data[1] << 8 | raw_data[0]);
    int16_t raw_acc_y = (int16_t)(raw_data[3] << 8 | raw_data[2]);
    int16_t raw_acc_z = (int16_t)(raw_data[5] << 8 | raw_data[4]);

    *x = (float)raw_acc_x * _accel_scale_factor;
    *y = (float)raw_acc_y * _accel_scale_factor;
    *z = (float)raw_acc_z * _accel_scale_factor;
    return 0;
}

int8_t BMI270::readGyroscope(float* x, float* y, float* z) {
    uint8_t raw_data[6]; 
    if (i2c_read(BMI270_GYR_DATA_X_LSB_ADDR, raw_data, 6) != 0) return -1;

    int16_t raw_gyr_x = (int16_t)(raw_data[1] << 8 | raw_data[0]);
    int16_t raw_gyr_y = (int16_t)(raw_data[3] << 8 | raw_data[2]);
    int16_t raw_gyr_z = (int16_t)(raw_data[5] << 8 | raw_data[4]);

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