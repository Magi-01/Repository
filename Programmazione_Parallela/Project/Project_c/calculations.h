#ifndef CALCULATIONS_H
#define CALCULATIONS_H

#include <stdint.h>
#include "ppm.h"

// Function to normalize a ray direction
void normalize_ray(float *ray);

// Function to find intersections with spheres
int intersections(PPMImage *image, float *origin, int num_spheres, float *direction, Sphere *spheres, uint32_t x, uint32_t y, uint32_t *color);

// Function to map pixel coordinates to viewport coordinates
void viewport_mapping(float x_pixel, float y_pixel, float *viewport_coords, PPMImage *image);

#endif
