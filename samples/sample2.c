int main() {
    int arr[5];
    int *p = arr;
    int *q = arr + 3;
    return q - p;
}
