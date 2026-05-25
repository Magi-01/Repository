
# Name: Fadhla Mohamed
# Sirname: Mutua
# Matricola: SM3201434

### Note 1:
WSL (windows subsytem for linux) decided to not work and due to time constraints I was forced to use the windows equivalento to `mmap`, `CreateFileMapping`, using the `windows.h` library.

It has the following prperties:

| Features | CreateFileMapping | mmap |
| -------- | ----------------- | ---- |
|File Handling | CreateFile | open |
| Resizing File| Implicit | ftruncate(fd, file_size) |
| Memory Mapping | CreateFileMapping + MapViewOfFile | mmap |
| Flushing Data | FlushViewOfFile | msync |
| Unmapping Memory | UnmapViewOfFile | munmap |
| Closing File | CloseHandle(hFile) | close(fd) |

Provided in the comments of `scene.h` is a more detailed explanation of how `CreateFileMapping` works in practice

### Note 2:
For both of the projects the input is done through the command line