#ifndef SCENE_H
#define SCENE_H

#include <stdint.h>
#include "ppm.h"

typedef struct {
    float center[3]; // x, y, z
    float radius;
} Sphere;

// Scene Rendering
float render_scene(PPMImage *image, Sphere *sphere, int num_spheres);

#endif
