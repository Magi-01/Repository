visited = {}
mydirection = {
        (0,-1):"s",
        (-1,0):"w",
        (1,0):"es",
        (0,1):"n"
}
for i in range(10):
    visited[i] = i*i
print(visited)
print(list(visited.values()))
k = [5,5]
m = [1,0]
z = (k[0]-m[0],k[1]-m[1])
tuple(z)
print(mydirection[z])