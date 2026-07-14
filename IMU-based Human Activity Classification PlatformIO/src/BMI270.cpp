#include <stdint.h>
#include "BMI270.h"
#include "bmi270_config.h"

extern void UART_printf(const char* format, ...);

#define BMI270_INIT_CTRL_ADDR      0x59
#define BMI270_INIT_DATA_ADDR      0x5E
#define BMI270_INTERNAL_STAT_ADDR  0x21

// --- Constructor ---
BMI270::BMI270(I2C_HandleTypeDef* hi2c) : _hi2c(hi2c),
                                         _device_address(BMI270_I2C_ADDR_PRIMARY),
                                         _accel_scale_factor(0.000061f * 9.80665f), 
                                         _gyro_scale_factor(0.061035f) {}

// --- Generic I2C read/write functions using HAL ---
int8_t BMI270::i2c_read(uint8_t reg_addr, uint8_t *data, uint16_t len) {
    if (HAL_I2C_Mem_Read(_hi2c, _device_address, reg_addr, I2C_MEMADD_SIZE_8BIT, data, len, 100) != HAL_OK) {
        UART_printf("I2C Read Error: Reg 0x%02X\r\n", reg_addr);
        return -1;
    }
    return 0;
}

int8_t BMI270::i2c_write(uint8_t reg_addr, uint8_t *data, uint16_t len) {
    if (HAL_I2C_Mem_Write(_hi2c, _device_address, reg_addr, I2C_MEMADD_SIZE_8BIT, data, len, 100) != HAL_OK) {
        UART_printf("I2C Write Error: Reg 0x%02X\r\n", reg_addr);
        return -1;
    }
    return 0;
}

int8_t BMI270::bmi270_set_command(uint8_t command) {
    return i2c_write(BMI270_CMD_ADDR, &command, 1);
}

// --- Initialize the BMI270 sensor ---
int8_t BMI270::init() {
    uint8_t chip_id = 0;
    UART_printf("Initializing BMI270...\r\n");
    // Try to read Chip ID
    if (i2c_read(BMI270_CHIP_ID_ADDR, &chip_id, 1) != 0 || chip_id != 0x24) {
        UART_printf("BMI270 Chip ID mismatch or read failed!\r\n");
        return -1;
    }
    // --- HARDWARE RECOVERY SEQUENCE ---
    // Forcefully disable power-saving mode first to ensure commands are accepted
    uint8_t pwr_conf_data = 0x00; 
    i2c_write(BMI270_PWR_CONF_ADDR, &pwr_conf_data, 1);
    HAL_Delay(5);
    // Explicitly write 0x00 to INIT_CTRL to completely terminate any previous aborted uploads
    uint8_t reset_init_ctrl = 0x00;
    i2c_write(BMI270_INIT_CTRL_ADDR, &reset_init_ctrl, 1);
    HAL_Delay(5);
    // Issue a fresh Soft Reset command to restore the internal ASIC registers
    uint8_t cmd_soft_reset = 0xB6;
    bmi270_set_command(cmd_soft_reset);
    HAL_Delay(50); // Give the chip plenty of time to clear its internal memory cache
    // Re-disable power-saving mode after soft reset
    i2c_write(BMI270_PWR_CONF_ADDR, &pwr_conf_data, 1);
    HAL_Delay(5);
    // 3. Prepare for configuration initialization
    uint8_t init_ctrl = 0x00;
    i2c_write(BMI270_INIT_CTRL_ADDR, &init_ctrl, 1);
    // Set internal memory pointer back to zero
    uint8_t init_addr_ptr[2] = {0x00, 0x00};
    i2c_write(0x5C, init_addr_ptr, 2); 
    HAL_Delay(1);
    // 4. Stream configuration file in 32-byte blocks
    uint16_t index = 0;
    while (index < 8192) {
        // Calculate the address pointer in words (each word is 2 bytes)
        uint16_t word_address = index / 2;
        
        // Prepare address register values (LSB and MSB offset)
        uint8_t addr_ptr[2];
        addr_ptr[0] = (uint8_t)(word_address & 0x0F);   // INIT_ADDR_0 (0x5B)
        addr_ptr[1] = (uint8_t)(word_address >> 4);     // INIT_ADDR_1 (0x5C)

        // Update the internal address pointers (0x5B starts the auto-increment)
        if (i2c_write(0x5B, addr_ptr, 2) != 0) {
            UART_printf("Failed to set address pointer at index %d\r\n", index);
            return -1;
        }

        // Stream 32 bytes to the initialization data register (0x5E)
        if (i2c_write(BMI270_INIT_DATA_ADDR, (uint8_t*)&bmi270_config_file[index], 32) != 0) {
            UART_printf("Config upload failed at byte %d\r\n", index);
            return -1;
        }

        index += 32;
        HAL_Delay(1); // Give the ASIC time to process the batch
    }
    // 5. Complete configuration initialization
    init_ctrl = 0x01;
    i2c_write(BMI270_INIT_CTRL_ADDR, &init_ctrl, 1);
    HAL_Delay(200);
    // 6. Verify initialization success
    uint8_t status = 0;
    i2c_read(BMI270_INTERNAL_STAT_ADDR, &status, 1);
    if ((status & 0x0F) != 0x01) {
        UART_printf("ASIC Initialization failed! Status: 0x%02X\r\n", status);
        return -1; 
    }
    UART_printf("ASIC Microcode loaded successfully!\r\n");
    // 7. Turn on the operational power engines
    uint8_t pwr_ctrl_data = 0x0E; // Enable Accelerometer, Gyroscope, and Temperature
    i2c_write(0x7D, &pwr_ctrl_data, 1);
    pwr_conf_data = 0x02; // Turn power saving mode back on
    i2c_write(BMI270_PWR_CONF_ADDR, &pwr_conf_data, 1);
    HAL_Delay(10);
    // 8. Configure sampling parameters (ODR, bandwidth)
    uint8_t acc_conf_data[] = {0xA8, 0x00}; 
    i2c_write(BMI270_ACC_CONF_ADDR, acc_conf_data, 2);
    uint8_t gyr_conf_data[] = {0xA8, 0x00}; 
    i2c_write(BMI270_GYR_CONF_ADDR, gyr_conf_data, 2);
    return 0;
}

// --- Read accelerometer data ---
int8_t BMI270::readAccelerometer(float* x, float* y, float* z) {
    uint8_t raw_data[6]; 
    if (i2c_read(BMI270_ACC_DATA_X_LSB_ADDR, raw_data, 6) != 0) return -1;
    int16_t raw_acc_x = (int16_t)((raw_data[1] << 8) | raw_data[0]);
    int16_t raw_acc_y = (int16_t)((raw_data[3] << 8) | raw_data[2]);
    int16_t raw_acc_z = (int16_t)((raw_data[5] << 8) | raw_data[4]);
    *x = (float)raw_acc_x * _accel_scale_factor;
    *y = (float)raw_acc_y * _accel_scale_factor;
    *z = (float)raw_acc_z * _accel_scale_factor;
    return 0;
}

// --- Read gyroscope data ---
int8_t BMI270::readGyroscope(float* x, float* y, float* z) {
    uint8_t raw_data[6]; 
    if (i2c_read(BMI270_GYR_DATA_X_LSB_ADDR, raw_data, 6) != 0) return -1;
    int16_t raw_gyr_x = (int16_t)((raw_data[1] << 8) | raw_data[0]);
    int16_t raw_gyr_y = (int16_t)((raw_data[3] << 8) | raw_data[2]);
    int16_t raw_gyr_z = (int16_t)((raw_data[5] << 8) | raw_data[4]);
    *x = (float)raw_gyr_x * _gyro_scale_factor;
    *y = (float)raw_gyr_y * _gyro_scale_factor;
    *z = (float)raw_gyr_z * _gyro_scale_factor;
    return 0;
}

// --- Read all sensor data at once ---
int8_t BMI270::readSensorData(bmi270_sensor_data_t* data) {
    if (readAccelerometer(&data->acc_x, &data->acc_y, &data->acc_z) != 0) return -1;
    if (readGyroscope(&data->gyr_x, &data->gyr_y, &data->gyr_z) != 0) return -1;
    return 0;
}