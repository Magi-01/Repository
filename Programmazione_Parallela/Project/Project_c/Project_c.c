#include "scene.h"  // Include scene.h, which already includes calculations.h
#include "ppm.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    const char *file_path = "./3d_image.ppm";

    Sphere spheres[] = {
        {{0.0f, 0.0f, 3.0f}, 1.0f}, // Sphere 1
        {{2.0f, 0.0f, 4.0f}, 1.0f}  // Sphere 2
    };

    PPMImage *image = pixel_parse(file_path); // Load the image

    render_scene(image, spheres, 2); // Render the scene

    free_ppm(image); // Free allocated memory
    return 0;
}
