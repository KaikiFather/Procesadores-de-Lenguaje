int main() {
    int values[3];
    values[0] = 1;
    values[1] = 2;
    values[2] = 3;
    int *p = values;
    int second = *(p + 1);
    int third = values[2];
    return second + third;
}
