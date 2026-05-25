/*
    Name: Fadhla Mohamed
    Sirname: Mutua
    Matricola: SM3201434
*/

#ifndef SCENE_H
#define SCENE_H

#include "ppm.h"

// Function to render the scene given viewport, objects, width, and height using mmap (CreateFileMapping)
void render_scene(char *output_file, PPMImage *image, float width, float height);

#endif
