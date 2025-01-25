#include <stdlib.h>
#include <stdint.h>
#include "calculations.h"  // Include calculations to use its functions
#include "ppm.h"
#include <stdio.h>
#include <math.h>

typedef struct {
    float center[3]; // x, y, z
    float radius;
} Sphere;


// Main function to render the scene
void render_scene(PPMImage *image, Sphere *spheres, int num_spheres) {
    float origin[3] = {0.0f, 0.0f, 0.0f}; // Camera position
    for (uint32_t y = 0; y < image->height; y++) {
        for (uint32_t x = 0; x < image->width; x++) {
            // Calculate viewport coordinates
            float viewport_coords[3];
            viewport_mapping((float)x, (float)y, viewport_coords, image->width, image->height);

            // Compute ray direction
            float direction[3] = {viewport_coords[0], viewport_coords[1], viewport_coords[2]};

            // Normalize the ray direction
            normalize_ray(direction);

            // Check for intersections with spheres
            intersections(origin, num_spheres, direction, spheres);
        }
    }
}