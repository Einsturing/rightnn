a = [([0] * 1024) for i in range(1024)]
for i in range(1024):
    if 0 <= i <= 31:
        if i - 1 >= 0:
            a[i][i - 1] = 1
        if i + 1 <= 31:
            a[i][i + 1] = 1
        if i + 31 >= 32:
            a[i][i + 31] = 1
        if i + 33 <= 63:
            a[i][i + 33] = 1
        a[i][i + 32] = 1
    elif 992 <= i <= 1023:
        if i - 1 >= 992:
            a[i][i - 1] = 1
        if i + 1 <= 1023:
            a[i][i + 1] = 1
        if i - 31 <= 991:
            a[i][i - 31] = 1
        if i - 33 >= 960:
            a[i][i - 33] = 1
        a[i][i - 32] = 1
    else:
        q = i // 32  # 1
        if i - 1 >= q * 32:
            a[i][i - 1] = 1
        if i + 1 <= q * 32 + 31:
            a[i][i + 1] = 1
        if i + 31 >= (q + 1) * 32:
            a[i][i + 31] = 1
        if i + 33 <= (q + 1) * 32 + 31:
            a[i][i + 33] = 1
        if i - 31 <= (q - 1) * 32 + 31:
            a[i][i - 31] = 1
        if(i - 33) >= (q - 1) * 32:
            a[i][i - 33] = 1
        a[i][i + 32] = 1
        a[i][i - 32] = 1

f = open('out.csv', 'w')
for i in range(1024):
    for j in range(1024):
        f.write(str(a[i][j]) + " ")
    f.write('\n')
f.close()
