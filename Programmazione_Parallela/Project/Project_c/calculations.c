#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "scene.h"
#include "ppm.h"

// Map screen coordinates to viewport coordinates
void viewport_mapping(float x_screen, float y_screen, float *viewport_coords, uint32_t width, uint32_t height) {
    viewport_coords[0] = (x_screen - width / 2.0f) / width;
    viewport_coords[1] = (height / 2.0f - y_screen) / height;
    viewport_coords[2] = 1.0f; // Fixed z for viewport
}

// Normalize a ray direction vector
void normalize_ray(float *direction) {
    float length = sqrtf(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    direction[0] /= length;
    direction[1] /= length;
    direction[2] /= length;
}

// Compute ray-sphere intersection
int ray_sphere_intersection(float *origin, float *direction, Sphere sphere, float *t) {
    float oc[3] = {origin[0] - sphere.center[0], origin[1] - sphere.center[1], origin[2] - sphere.center[2]};
    float a = direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
    float b = 2.0f * (oc[0] * direction[0] + oc[1] * direction[1] + oc[2] * direction[2]);
    float c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - sphere.radius * sphere.radius;

    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return 0; // No intersection
    }
    *t = (-b - sqrtf(discriminant)) / (2.0f * a);
    return 1; // Intersection found
}

// Process intersections and shade the pixel
void intersections(PPMImage *image, float *origin, int num_spheres, float *direction, Sphere *spheres) {
    float closest_t = INFINITY; // Initialize to the largest float value
    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (ray_sphere_intersection(origin, direction, spheres[i], &t)) {
            if (t < closest_t) {
                closest_t = t;
                // Shade the pixel based on the sphere's properties
                image->data[(int)((origin[1] * image->width + origin[0]) * 3 + 0)] = 255; // R
                image->data[(int)((origin[1] * image->width + origin[0]) * 3 + 1)] = 0;   // G
                image->data[(int)((origin[1] * image->width + origin[0]) * 3 + 2)] = 0;   // B
            }
        }
    }
}
