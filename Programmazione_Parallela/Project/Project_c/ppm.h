#ifndef PPM_H
#define PPM_H

#include <stdint.h>

// Define the PPMImage structure
typedef struct __attribute__((packed)) {
    char magic_number[2]; // PPM format identifier (e.g., "P6")
    uint32_t width;       // Image width
    uint32_t height;      // Image height
    uint32_t max_color;   // Max color intensity
    uint8_t *data;        // Pointer to pixel data (RGB format)
} PPMImage;

PPMImage *pixel_parse(file_path);

void free_ppm(image);

#endif