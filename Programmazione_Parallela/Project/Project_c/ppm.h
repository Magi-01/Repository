#ifndef PPM_H
#define PPM_H

#include <stdint.h>

typedef struct {
    float center[3];   // Sphere position (x, y, z)
    float radius;        // Radius of the sphere
    uint8_t color[3];    // Color of the sphere (RGB format)
} Sphere;


// Define the PPMImage structure with packed attribute
typedef struct __attribute__((packed)) {
    char magic_number[2]; // PPM format identifier, e.g., "P6"
    uint32_t viewport[3]; // Viewport size (x, y, z)
    uint32_t background[3]; // Background color (RGB)
    uint32_t OBJ_N; // Number of objects (spheres)
    Sphere *spheres; // Array of spheres
    uint32_t *data;
} PPMImage;

// Function to parse the PPM file
PPMImage parse_ppm(const char *file_path);

// Function to save the PPM image to a file
void save_ppm_image(const char *file_path, PPMImage *image);

#endif
