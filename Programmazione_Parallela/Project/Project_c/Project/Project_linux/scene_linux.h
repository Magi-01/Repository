/*
    Name: Fadhla Mohamed
    Sirname: Mutua
    Matricola: SM3201434
*/

#ifndef SCENE_LINUX_H
#define SCENE_LINUX_H

#include "ppm_linux.h"

// Function to render the scene given viewport, objects, width, and height using mmap (CreateFileMapping)
void render_scene(char *output_file, PPMImage *image, float width, float height);

#endif
