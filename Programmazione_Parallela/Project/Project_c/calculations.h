#ifndef CALCULATIONS_H
#define CALCULATIONS_H

#include <math.h> // For mathematical operations
#include "ppm.h"

void viewport_mapping(float x_screen, float y_screen, float *viewport_coords, uint32_t width, uint32_t height);

int ray_sphere_intersection(float *origin, float *direction, Sphere sphere, float *t);

float* normalize_ray(float* direction);

void intersections(float* origin, int num_spheres, float *direction, Sphere spheres);

#endif