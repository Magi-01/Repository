#include "calculations.h"
#include "scene.h"
#include "ppm.h"
#include <math.h>
#include <float.h>

#define EPSILON 0.00001f

// Function to normalize a ray direction
void normalize_ray(float *ray) {
    float length = sqrt(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);
    if (length > EPSILON) {
        ray[0] /= length;
        ray[1] /= length;
        ray[2] /= length;
    }
}

// Function to check for ray-sphere intersection
int ray_sphere_intersection(float *origin, float *direction, Sphere *sphere, float *t) {
    float oc[3] = {origin[0] - sphere->center[0], origin[1] - sphere->center[1], origin[2] - sphere->center[2]};
    float a = direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
    float b = 2.0f * (oc[0] * direction[0] + oc[1] * direction[1] + oc[2] * direction[2]);
    float c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - sphere->radius * sphere->radius;

    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return 0; // No intersection
    }

    float sqrt_discriminant = sqrt(discriminant);
    float t0 = (-b - sqrt_discriminant) / (2.0f * a);
    float t1 = (-b + sqrt_discriminant) / (2.0f * a);

    if (t0 > t1) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }

    *t = t0 > EPSILON ? t0 : t1; // Use the smaller valid t value
    return 1; // Intersection found
}

// Function to find intersections with spheres
int intersections(PPMImage *image, float *origin, int num_spheres, float *direction, Sphere *spheres, uint32_t x, uint32_t y, uint32_t *color) {
    float closest_t = FLT_MAX;
    int hit_sphere_index = -1;

    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (ray_sphere_intersection(origin, direction, &image->spheres[i], &t)) {
            if (t < closest_t) {
                closest_t = t;
                hit_sphere_index = i;
            }
        }
    }

    if (hit_sphere_index != -1) {
        color[0] = spheres[hit_sphere_index].color[0];
        color[1] = spheres[hit_sphere_index].color[1];
        color[2] = spheres[hit_sphere_index].color[2];
        return 1;
    }

    color[0] = image->background[0]; // Background color
    color[1] = image->background[1];
    color[2] = image->background[2];
    return 0;
}

// Function to map pixel coordinates to viewport coordinates
void viewport_mapping(float x_pixel, float y_pixel, float *viewport_coords, PPMImage *image) {
    float aspect_ratio = image->viewport[0];
    float viewport_width = image->viewport[2];
    float viewport_height = image->viewport[1];

    viewport_coords[0] = (2.0f * x_pixel / (float)(viewport_width - 1)) - 1.0f; // x in [-1, 1]
    viewport_coords[1] = (2.0f * y_pixel / (float)(viewport_height - 1)) - 1.0f; // y in [-1, 1]
    viewport_coords[2] = 1.0f; // z = 1, in front of the camera

    // Scale by the aspect ratio
    viewport_coords[0] *= aspect_ratio;
    viewport_coords[1] *= (1.0f / viewport_height);  // Adjust for height
}
