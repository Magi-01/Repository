/*
    Name: Fadhla Mohamed
    Sirname: Mutua
    Matricola: SM3201434
*/

#ifndef PPM_H
#define PPM_H

#include <stdint.h>

/*
Define the Sphere structure:
float center[3] --> Sphere position (x, y, z),
float radius --> Radius of the sphere,
uint8_t color[3] --> Color of the sphere (RGB format)
*/
typedef struct {
    float center[3];
    float radius;
    uint8_t color[3];
} Sphere;


/*
Define the PPMImage structure with packed attribute:
Char Format[2] --> (e.g. "P6"),
float Viewport[3] --> (x,y,z) scaling,
uint8_t background[3] --> Background color (RGB),
Int ONJ_N --> Number of objects,
Sphere struct --> Array of spheres
*/
typedef struct __attribute__((packed)) {
    char magic_number[2];
    float viewport[3];
    uint8_t background[3];
    int OBJ_N;
    Sphere *spheres;
} PPMImage;

// Function to parse the PPM file
PPMImage *parse_ppm(const char *file_path, const float width, const float height);

// Function to check file values
void print_ppm_data(PPMImage *image);

#endif
