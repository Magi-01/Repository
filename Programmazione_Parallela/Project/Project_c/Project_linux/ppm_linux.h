/*
    Name: Fadhla Mohamed
    Sirname: Mutua
    Matricola: SM3201434
*/

#ifndef PPM_LINUX_H
#define PPM_LINUX_H

#include <stdint.h>

/*
    Sphere struct: holds all spheres in the scene.
    Packed attribute ensures no padding between members.
    - cx, cy, cz: arrays of sphere centers (x, y, z)
    - r: array of radii
    - color: array of RGB colors (uint8_t[3])
    - n: number of spheres
*/
typedef struct __attribute__((packed)){
    float *cx, *cy, *cz;       // arrays of x, y, z
    float *r;                   // radii
    uint8_t (*color)[3];        // colors
    int n;
} Sphere;


/*
    PPMImage struct: represents the scene and image info
    - magic_number: e.g. "P6"
    - viewport: 3 floats representing viewport dimensions (x,y,z)
    - background: RGB background color
    - OBJ_N: number of spheres
    - spheres: the Sphere struct with all objects
*/
typedef struct __attribute__((packed)) {
    char magic_number[3];
    float viewport[3];
    uint8_t background[3];
    int OBJ_N;
    Sphere spheres;
} PPMImage;

// Function to parse the PPM file
PPMImage *parse_ppm(const char *file_path);

// Function to check file values
void print_ppm_data(PPMImage *image);

void free_image(PPMImage *image);

#endif
