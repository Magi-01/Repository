#include "calculations.h"
#include "scene.h"
#include "ppm.h"
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <omp.h>

// Function to normalize a ray direction
void normalize_ray(float *ray) {
    float length = sqrt(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);
    if (length > 0) {
        ray[0] /= length;
        ray[1] /= length;
        ray[2] /= length;
    }
}

// Function to compute the dot product of two vectors
float dot(const float *v1, const float *v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// Function to check for ray-sphere intersection
int ray_sphere_intersection(const float *origin, const float *direction, const Sphere *sphere, float *t, int x, int y) {
    float oc[3] = {
        sphere->center[0] - origin[0],
        sphere->center[1] - origin[1],
        sphere->center[2] - origin[2]
    };
    //printf("(x,y): (%d,%d)\n", x, y);
    //printf("sphere - centre: (%f,%f,%f)\n", oc[0],oc[1],oc[2]);
    //printf("direction (%f, %f, %f)", direction[0], direction[1], direction[2]);
    float a = dot(direction, direction);
    float b = -2.0f * dot(oc, direction);
    float c = dot(oc, oc) - sphere->radius * sphere->radius;
    float discriminant = b * b - 4 * a * c;
    int count = 0;
    //printf("discriminant: %f\n", discriminant);
    if (discriminant < 0) {
        return 0;  // No intersection
    }

    float sqrt_d = sqrtf(discriminant);
    float t0 = (-b - sqrt_d) / (2.0f * a);
    float t1 = (-b + sqrt_d) / (2.0f * a);
    //printf("(t0,t1): (%f,%f)\n\n", t0, t1);
    // Find the nearest positive t
    if (t0 > 0 && t1 > 0) {
        *t = fminf(t0, t1);
        return 1;
    } else if (t0 >= 0) {
        *t = t0;
        return 1;
    } else if (t1 > 0) {
        *t = t1;
        return 1;
    }

    return 0;  // Intersection is behind the ray origin
}

// Function to find intersections with spheres
int intersections(PPMImage *image, float *origin, int num_spheres, float *direction, Sphere *spheres, int x, int y, uint8_t *color) {
    float closest_t = FLT_MAX;
    int hit_sphere_index = -1;

    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (ray_sphere_intersection(origin, direction, &spheres[i], &t, x, y)) {
            if (t < closest_t) {
                closest_t = t;
                hit_sphere_index = i;
            }
        }
    }


    if (hit_sphere_index != -1) {
        // A sphere was hit; return its color
        color[0] = spheres[hit_sphere_index].color[0];
        color[1] = spheres[hit_sphere_index].color[1];
        color[2] = spheres[hit_sphere_index].color[2];
        return 1;
    }

    // No intersection; return background color
    color[0] = image->background[0];
    color[1] = image->background[1];
    color[2] = image->background[2];
    return 0;
}

// Function to map pixel coordinates to viewport coordinates
void viewport_mapping(float x_pixel, float y_pixel, float *viewport_coords, PPMImage *image) {
    float aspect_ratio_x = image->viewport[0];
    float aspect_ratio_y = image->viewport[1];

    viewport_coords[0] = (2.0f * x_pixel / (1920.0f - 1.0f) - 1.0f) * aspect_ratio_x;
    viewport_coords[1] = (2.0f * y_pixel / (1080.0f - 1.0f) - 1.0f) * aspect_ratio_y;
    viewport_coords[2] = 1.0f;
}
