int main() {
    int x = 5;
    int y = -x;
    int z = !x;
    if (y + z < 0) {
        z = 1;
    }
    return y + z;
}
