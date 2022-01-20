import numpy as np
import pandas as pd

def giaithua(a):
    if a <= 1:
        return 1
    else:
        return a*giaithua(a-1)

def chiadoi(f, a, b, n = 0, epxilon = 0):
    if n != 0:
        for i in range(n):
            print(f'a: {a}, b: {b}')
            x = (a+b)/2
            y = f(x)
            if y == 0:
                print(x, 'delta: 0')
                break
            if(y*f(a) < 0):
                b = x
            else:
                a = x
            delta_x = (b - a)
            print(f'x: {x}\nf(x) = {y}\ndelta: {delta_x}\n')
    if epxilon != 0:
        l = a
        r = b
        n = int(1 + np.log2((r - l)/epxilon))
        print('n =',n,'\n')
        for i in range(n):
            print(f'a: {l}, b: {r}')
            x = (l+r)/2
            y = f(x)
            if y == 0:
                delta_x = 0
                break
            else:
                if y*f(l) < 0:
                    r = x
                else:
                    l = x
                delta_x = (r - l)
            print(f'x: {x}\nf(x) = {y}\ndelta x: {delta_x}\n')


def daycung(f, daoham1, a, b, n=0, epxilon=0, mode='err'):
    m1 = min(np.abs(daoham1(a)), np.abs(daoham1(b)))
    M1 = max(np.abs(daoham1(a)), np.abs(daoham1(b)))
    print(f'f(a) = {f(a)},  f(b) = {f(b)}')
    print(f'm1 = {m1}\nM1 = {M1}')
    t = (a * f(b) - b * f(a)) / (f(b) - f(a))
    if f(t) * f(a) > 0:
        x = a
        d = b
    else:
        x = b
        d = a
    print(f'x0: {x}\nd: {d}\n')
    delta_x = epxilon + 1
    err = epxilon + 1
    if epxilon != 0:
        c = 0
        if mode == 'delta_x':
            while delta_x > epxilon:
                x_pre = x
                y = f(x)
                x = x - y * (x - d) / (y - f(d))
                delta_x = np.abs(f(x)) / m1
                err = (M1 - m1) * np.abs(x - x_pre) / m1
                print(f'x: {x}\nxấp xỉ liên tiếp: {err}\ndelta_x: {delta_x}\n')
                c += 1

        if mode == 'err':
            while err > epxilon:
                x_pre = x
                y = f(x)
                x = x - y * (x - d) / (y - f(d))
                delta_x = np.abs(f(x)) / m1
                err = (M1 - m1) * np.abs(x - x_pre) / m1
                print(f'x: {x}\nxấp xỉ liên tiếp: {err}\ndelta_x: {delta_x}\n')
                c += 1
        print(f'n = {c}')
    else:
        while n > 0:
            x_pre = x
            y = f(x)
            x = x - y * (x - d) / (y - f(d))
            delta_x = np.abs(f(x)) / m1
            err = (M1 - m1) * np.abs(x - x_pre) / m1
            print(f'x: {x}\nxấp xỉ liên tiếp: {err}\ndelta_x: {delta_x}\n')
            n -= 1
    print(f'epxilon0: {(m1 * epxilon) / (M1 - m1)}')

def tieptuyen(f, daoham1, daoham2, a, b, n=0, epxilon=0, mode='err'):
    m1 = min(np.abs(daoham1(a)), np.abs(daoham1(b)))
    M2 = max(np.abs(daoham2(a)), np.abs(daoham2(b)))
    print(f'f(a) = {f(a)},  f(b) = {f(b)}')
    print(f'm1 = {m1}\nM2 = {M2}')
    t = (a*f(b) - b*f(a)) / (f(b) - f(a))
    if f(t)*f(a) < 0:
        x = a
    else:
        x = b
    print(f'x0: {x}\n')
    if epxilon != 0:
        c = 0
        delta_x = epxilon + 1
        err = epxilon + 1
        if mode=='err':
            while err > epxilon:
                a = x
                b = x - f(x)/daoham1(x)
                x = x - f(x)/daoham1(x)
                delta_x = np.abs(f(x))/m1
                err = np.abs(b - a)**2*M2/(2*m1)
                print(f'x: {x}\nxấp xỉ liên tiếp: {err}\ndelta x: {delta_x}\n')
                c += 1
        if mode=='delta_x':
            while delta_x > epxilon:
                a = x
                b = x - f(x)/daoham1(x)
                x = x - f(x)/daoham1(x)
                delta_x = np.abs(f(x))/m1
                err = np.abs(b - a)**2*M2/(2*m1)
                print(f'x: {x}\nxấp xỉ liên tiếp: {err}\ndelta x: {delta_x}\n')
                c += 1
        print(f'n = {c}')
    if n != 0:
        while n > 0:
            a = x
            b = x - f(x)/daoham1(x)
            x = x - f(x)/daoham1(x)
            delta_x = np.abs(f(x))/m1
            err = np.abs(b - a)**2*M2/(2*m1)
            print(f'x: {x}\nxấp xỉ liên tiếp: {err}\ndelta x: {delta_x}\n')
            n-=1
    print(f'epxilon0: {np.sqrt(2*m1*epxilon/M2)}')

def lapdon(f, phi, x0, q, epxilon=0, n=0):
    print(f'q = {q}\n')
    x = x0
    if epxilon != 0:
        c = 0
        err = epxilon + 1
        while err > epxilon:
            x_pre = x
            x = phi(x_pre)
            err = q*np.abs(x - x_pre) / (1 - q)
            print(f'x: {x}\nxấp xỉ liên tiếp: {err}\n')
            c+=1
        print(f'n = {c}')
    if n != 0:
        for i in range(n):
            x_pre = x
            x = phi(x_pre)
            err = q*np.abs(x - x_pre) / (1 - q)
            print(f'x: {x}\nxấp xỉ liên tiếp: {err}\n')
    print(f'epxilon0: {(1-q)*epxilon/q}')

def khu(a, b, ind):
    b = b - b[ind]/a[ind] * a
    return b

def Gauss(a):
    print(a, '\n')
    for i in range(len(a)):
        j = i
        for r in range(j, len(a[i])):
            if a[i, j] == 0:
                check = 0
                temp = a[i].copy()
                for k in range(i + 1, len(a)):
                    if a[k, j] != 0:
                        a[i] = a[k]
                        a[k] = temp
                        check = 1
                        print(a,'\n')
                        break
                if check == 0:
                    j += 1
        for l in range(i + 1, len(a)):
            a[l] = khu(a[i], a[l], j)
        print(a,'\n')
    return a

def giai_Gauss(a):
    check = []
    nghiem = np.zeros((len(a), 1))
    for i in range(len(a)-1, -1, -1):
        nghiem[i] = a[i, -1]
        for j in check:
            nghiem[i] -= nghiem[j]*a[i, j]
        nghiem[i] /= a[i, i]
        check.append(i)
    return nghiem

def Gauss_Jordan(a):
    print(a, '\n')
    ind = list()
    check = []
    while True:
        print("đã khử hàng: ", ind)
        print("chọn phần tử khử: ")
        i = int(input("hàng: ")) - 1
        j = int(input('cột: ')) - 1
        check.append([i, j])
        ind.append(i + 1)
        for k in range(len(a)):
            if k != i:
                a[k] = khu(a[i], a[k], j)
        for l in range(len(a)):
            for m in range(len(a[l])):
                if np.abs(a[l, m]) < 1 * (10 ** -10):
                    a[l, m] = 0
        print(a, '\n')
        if len(ind) == len(a) - 1:
            break
        select = input("khử tiếp không (y/n)?")
        if select == 'n':
            break
    for i in check:
        if a[i[0], i[1]] != 1:
            a[i[0]] = a[i[0]] / a[i[0], i[1]]
    print(a)
    return a

def chuanvocung(a):# chuẩn hàng
    lst = []
    for i in a:
        lst.append(np.sum(np.abs(i)))
    return max(lst)

def chuan1(a):# chuẩn cột
    if len(a[0]) != 1:
        lst = []
        for i in range(len(a)):
            lst.append(np.sum(np.abs(a[:,i])))
        return max(lst)
    else:
        return np.sum(np.abs(a))

def chuan2(a):
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(a.T, a))
    return np.max(np.sqrt(eigenvalues))

def giai_GaussJordan(a):
    nghiem = np.zeros((len(a), 1))
    m = []
    l = []
    for i in range(len(a)):
        if len(np.where(a[i] == 0)[0]) == len(a[i]) - 2:
            for j in range(len(a[i]) - 1):
                if a[i, j] != 0:
                    m.append(j)
                    l.append(i)
                    nghiem[j] = a[i, -1] / a[i, j]
                    break

    for i in range(len(a)):
        if i not in l:
            for j in range(len(a[i]) - 1):
                if a[i, j] == 1:
                    for k in m:
                        if a[i, k] != 0 and a[i, k] != 1:
                            nghiem[j] = (a[i, -1] - nghiem[k] * a[i, k]) / a[i, j]

    return nghiem

def lapdon_matrix(B, d, X0, q, n=0, epxilon=0, mode='chuanhang', lamda = 0):
    X = X0.reshape((-1,1)).copy()
    d = d.reshape((-1,1))
    if mode == 'chuanhang':
        print(f'q = {q}\n')
        if n != 0:
            while n != 0:
                X_pre = X.copy()
                X = np.dot(B, X_pre) + d
                print(f'X =\n {X}')
                err = np.max(np.abs(X - X_pre))*q/(1-q)
                print(f'xấp xỉ liên tiếp: {err}\n')
                n -= 1
        if epxilon != 0:
            err = epxilon + 1
            c = 0
            while err > epxilon:
                X_pre = X.copy()
                X = np.dot(B, X_pre) + d
                print(f'X =\n {X}')
                err = np.max(np.abs(X - X_pre))*q/(1-q)
                print(f'xấp xỉ liên tiếp: {err}\n')
                c += 1
            print(f'n = {c}')
    if mode == 'chuancot':
        print(f'q = {q}\n')
        if n != 0:
            while n != 0:
                X_pre = X.copy()
                X = np.dot(B, X_pre) + d
                print(f'X =\n {X}')
                err = lamda * chuan1(X - X_pre) * q/(1-q)
                print(f'xấp xỉ liên tiếp: {err}\n')
                n -= 1
        if epxilon != 0:
            err = epxilon + 1
            c = 0
            while err > epxilon:
                X_pre = X.copy()
                X = np.dot(B, X_pre) + d
                print(f'X =\n {X}')
                err = lamda * chuan1(X - X_pre) * q/(1-q)
                print(f'xấp xỉ liên tiếp: {err}\n')
                c += 1
            print(f'n = {c}')

def cheotroihang(a):
    for i in range(len(a)):
        tong = 0
        for j in range(len(a[i])):
            if j != i:
                tong += np.abs(a[i, j])
        if np.abs(a[i, i]) < tong:
            return False
    return True

def cheotroicot(a):
    for i in range(len(a)):
        tong = 0
        for j in range(len(a)):
            if j != i:
                tong += np.abs(a[j, i])
        if np.abs(a[i, i]) < tong:
            return False
    return True

def lapJacobi(B, d, X0, n=0, epxilon=0):
    lst = []
    d = d.reshape((-1, 1))
    if cheotroihang(B):
        print('chéo trội hàng')
        for i in range(len(B)):
            lst.append(B[i, i])
            d[i] = d[i] / B[i, i]
            B[i] = B[i] / B[i, i]
            B[i] *= -1
            B[i, i] = 0
        print(f'B = \n{B}\nd = \n{d}\n')
        q = chuanvocung(B)
        lapdon_matrix(B, d, X0, q, n=n, epxilon=epxilon, mode='chuanhang')

    if cheotroicot(B):
        print('chéo trội cột')
        C = B.copy().T
        for i in range(len(B)):
            lst.append(np.abs(B[i, i]))
            d[i] = d[i] / B[i, i]
            B[i] = B[i] / B[i, i]
            B[i] *= -1
            B[i, i] = 0
        for i in range(len(C)):
            C[i] = C[i] / C[i, i]
            C[i] *= -1
            C[i, i] = 0
        print(f'B = \n{B}\nd = \n{d}\n')
        q = chuanvocung(C)
        lamda = max(lst) / min(lst)
        print(f'lamda: {lamda}')
        lapdon_matrix(B, d, X0, q, n=n, epxilon=epxilon, mode='chuancot', lamda=lamda)

def hoocner(a,b):
    b = b.reshape(-1,1)
    zero = np.zeros((1, 1))
    c = np.concatenate([zero, b])
    result = a.copy()
    final = a.copy()
    for i in range(len(b)):
        for j in range(1, len(result[0])):
            result[0,j] = result[0, j-1]*b[i,0] + a[0,j]
            if np.abs(result[0,j]) < 10**(-10):
                result[0,j] = 0
        final = np.concatenate([final, result])
    final = np.concatenate([c, final], axis=1)
    return final

def hoocnenguoc(a):
    final = np.array([[1, -a[0,0]]])
    for i in range(1, len(a[0])):
        x = a[0,i]
        zero = np.zeros((len(final), 1))
        final = np.concatenate([zero, final], axis=1)
        hang2 = np.zeros((1, len(final[-1])))
        hang1 = final[-1,:]*x
        hang2[0,0] = 1
        for i in range(1, len(hang1) - 1):
            hang2[0,i] = final[-1,i+1] - hang1[i]
            if np.abs(hang2[0,i+1]) < 10**(-10):
                hang2[0,i+1] = 0
        hang2[0, -1] = 0 - hang1[-1]
        final = np.concatenate([final,hang2])
    return final

def Dy(a):
    multi = np.ones((a.shape[1],1))
    result = np.zeros(a.shape)
    final = np.zeros(a.shape)
    for i in range(len(a[0])):
        for j in range(len(a[0])):
            if i == j:
                result[0,j] = 1
            else:
                result[0,j] = a[0,i] - a[0,j]
            multi[i, 0] *= result[0,j]
        final = np.append(final, result, axis=0)
    final = np.concatenate([final[1:], multi], axis=1)
    return final

def DathucLGcoban(final2, Dy_a):
    heso = final2[1:, 1:].copy()
    for i in range(len(heso)):
        heso[i] = heso[i] / Dy_a[i, -1]
    return heso[:,:-1]

def NoisuyLG(A, Y, b):
    final1 = hoocnenguoc(A)
    print('\nSơ đồ nhân:\n', pd.DataFrame(final1, index=A[0], columns=[i for i in range(len(A[0]), -1, -1)]), "\n")

    final2 = hoocner(np.array([final1[-1]]), A)
    print('Sơ đồ chia:\n', pd.DataFrame(final2[1:, 1:], index=final2[1:, 0], columns=final2[0, 1:]))

    Dy_a = Dy(A)
    print('\nDi:\n', pd.DataFrame(Dy_a))

    Di = Dy_a[:, -1].copy().T
    print("\nY/Di =\n", Y / Di)

    hesoLGcoban = DathucLGcoban(final2, Dy_a)
    print('\nHệ số đa thức Lagrange cơ bản\n', pd.DataFrame(hesoLGcoban))

    super_final = np.dot(Y, hesoLGcoban)

    print('\nHệ số đa thức P tìm được:\n',
          pd.DataFrame(super_final, columns=[str(i) for i in range(len(super_final[0]) - 1, -1, -1)]))

    final = hoocner(super_final, b.astype(np.float))

    print(f'\nTính P tại x = {b[0]}\n',
          pd.DataFrame(final[:, 1:], index=final[:, 0], columns=[str(i) for i in range(len(final[0]) - 2, -1, -1)]))

    print(f'\n{final[-1, -1]}')


def bangTSP(X, Y):
    final = np.concatenate([X.T, Y.T], axis=1)

    for i in range(len(X[0]) - 1):
        TSP = np.zeros((len(X[0]), 1))
        for j in range(i + 1, len(X[0])):
            TSP[j, 0] = (final[j, -1] - final[j - 1, -1]) / (final[j, 0] - final[j - (i + 1), 0])
            if np.abs(TSP[j, 0]) < 10 ** (-10):
                TSP[j, 0] = 0
        final = np.concatenate([final, TSP], axis=1)

    return final


def NoisuyNewtonTien(X, Y, b):
    TSP = bangTSP(X, Y)
    print('\nBảng tỷ sai phân:\n', pd.DataFrame(TSP[:, 1:], index=TSP[:, 0]))

    A = X[:, :-1].copy()
    bangnhan = hoocnenguoc(A)
    A = A.T
    bangnhan = np.concatenate([A, bangnhan], axis=1)
    zero = np.zeros((1, X.shape[1] + 1))
    zero[0, -1] = 1
    delbiet = np.concatenate([zero, bangnhan])

    print('\nBảng nhân:\n',
          pd.DataFrame(delbiet[:, 1:], index=delbiet[:, 0], columns=[i for i in range(len(X[0]) - 1, -1, -1)]))

    giatriTSP = []
    for i in range(len(TSP)):
        for j in range(len(TSP[i])):
            if j == i + 1:
                giatriTSP.append(TSP[i, j])
    super_final = np.array([np.dot(giatriTSP, delbiet[:, 1:])])
    print('\nHệ số đa thức P tìm được:\n',
          pd.DataFrame(super_final, columns=[str(i) for i in range(len(super_final[0]) - 1, -1, -1)]))

    final = hoocner(super_final, b.astype(np.float))
    print(f'\nTính P tại x = {b[0]}\n',
          pd.DataFrame(final[:, 1:], index=final[:, 0], columns=[str(i) for i in range(len(final[0]) - 2, -1, -1)]))

    print(f'\n{final[-1, -1]}')


def NoisuyNewtonLui(X, Y, b):
    TSP = bangTSP(X, Y)
    print('\nBảng tỷ sai phân:\n', pd.DataFrame(TSP[:, 1:], index=TSP[:, 0]))

    X = np.flip(X)
    A = X[:, :-1].copy()
    bangnhan = hoocnenguoc(A)
    A = A.T
    bangnhan = np.concatenate([A, bangnhan], axis=1)
    zero = np.zeros((1, X.shape[1] + 1))
    zero[0, -1] = 1
    delbiet = np.concatenate([zero, bangnhan])

    print('\nBảng nhân:\n',
          pd.DataFrame(delbiet[:, 1:], index=delbiet[:, 0], columns=[i for i in range(len(X[0]) - 1, -1, -1)]))

    giatriTSP = TSP[-1, 1:]

    super_final = np.array([np.dot(giatriTSP, delbiet[:, 1:])])
    print('\nHệ số đa thức P tìm được:\n',
          pd.DataFrame(super_final, columns=[str(i) for i in range(len(super_final[0]) - 1, -1, -1)]))

    final = hoocner(super_final, b.astype(np.float))
    print(f'\nTính P tại x = {b[0]}\n',
          pd.DataFrame(final[:, 1:], index=final[:, 0], columns=[str(i) for i in range(len(final[0]) - 2, -1, -1)]))

    print(f'\n{final[-1, -1]}')


def bangSP(X, Y):
    final = np.concatenate([X.T, Y.T], axis=1)

    for i in range(len(X[0]) - 1):
        TSP = np.zeros((len(X[0]), 1))
        for j in range(i + 1, len(X[0])):
            TSP[j, 0] = (final[j, -1] - final[j - 1, -1])
            if np.abs(TSP[j, 0]) < 10 ** (-10):
                TSP[j, 0] = 0
        final = np.concatenate([final, TSP], axis=1)

    return final

def NoisuyNTcachdeuTien(X, Y, b):
    h = (X[0, -1] - X[0, 0]) / (len(X[0]) - 1)
    print(f"\nh = {h}")

    SP = bangSP(X, Y)
    print('\nBảng sai phân:\n', pd.DataFrame(SP[:, 1:], index=SP[:, 0]))

    heso_bangnhan = np.linspace(0, len(X[0]) - 2, len(X[0]) - 1).reshape(1, -1)
    index = heso_bangnhan[0].copy()
    khong = np.array([0])
    index = np.concatenate([khong, index])

    bangnhan = hoocnenguoc(heso_bangnhan)
    zero = np.zeros((1, X.shape[1]))
    zero[0, -1] = 1
    bangnhan_final = np.concatenate([zero, bangnhan])
    print('\nBảng nhân:\n',
          pd.DataFrame(bangnhan_final, index=index, columns=[i for i in range(len(X[0]) - 1, -1, -1)]))

    giatriSP = []
    for i in range(len(SP)):
        for j in range(len(SP[i])):
            if j == i + 1:
                giatriSP.append(SP[i, j])
    giatriSP = np.array([giatriSP])
    heso_Pn = []
    for i in range(len(giatriSP[0])):
        heso_Pn.append(giatriSP[0, i] / giaithua(i))
    heso_Pn = np.array([heso_Pn])
    print(f"\nSau khi chia giai thừa:\n", pd.DataFrame(heso_Pn))

    super_final = np.dot(heso_Pn, bangnhan_final)
    print('\nHệ số đa thức Pn(t):\n', pd.DataFrame(super_final, columns=[i for i in range(len(X[0]) - 1, -1, -1)]))

    t = (b - X[0, 0]) / h
    print(f'\nt = {t[0]}\n')
    Pn_bt = hoocner(super_final, t)
    print(f'\nTính P tại t = {t[0]}\n',
          pd.DataFrame(Pn_bt[:, 1:], index=Pn_bt[:, 0], columns=[str(i) for i in range(len(Pn_bt[0]) - 2, -1, -1)]))

    print(f'\n{Pn_bt[-1, -1]}')


def NoisuyNTcachdeuLui(X, Y, b):
    h = (X[0, -1] - X[0, 0]) / (len(X[0]) - 1)
    print(f"\nh = {h}\n")

    SP = bangSP(X, Y)
    print('Bảng sai phân:\n', pd.DataFrame(SP[:, 1:], index=SP[:, 0]))

    heso_bangnhan = -np.linspace(0, len(X[0]) - 2, len(X[0]) - 1).reshape(1, -1)
    index = np.array([heso_bangnhan[0].copy()])
    khong = np.array([[0]])
    index = np.concatenate([khong, index], axis=1)

    bangnhan = hoocnenguoc(heso_bangnhan)
    zero = np.zeros((1, X.shape[1]))
    zero[0, -1] = 1
    bangnhan_final = np.concatenate([zero, bangnhan])
    print('\nBảng nhân:\n',
          pd.DataFrame(bangnhan_final, index=index[0], columns=[i for i in range(len(X[0]) - 1, -1, -1)]))

    giatriSP = np.array([SP[-1, 1:].copy()])

    heso_Pn = []
    for i in range(len(giatriSP[0])):
        heso_Pn.append(giatriSP[0, i] / giaithua(i))
    heso_Pn = np.array([heso_Pn])
    print(f"\nSau khi chia giai thừa:\n", pd.DataFrame(heso_Pn))

    super_final = np.dot(heso_Pn, bangnhan_final)
    print('\nHệ số đa thức Pn(t):\n', pd.DataFrame(super_final, columns=[i for i in range(len(X[0]) - 1, -1, -1)]))

    t = (b - X[0, -1]) / h
    print(f'\nt = {t[0]}\n')
    Pn_bt = hoocner(super_final, t)
    print(f'\nTính P tại t = {t[0]}\n',
          pd.DataFrame(Pn_bt[:, 1:], index=Pn_bt[:, 0], columns=[str(i) for i in range(len(Pn_bt[0]) - 2, -1, -1)]))

    print(f'\n{Pn_bt[-1, -1]}')

def nhapX(somoc):
    X = []
    for i in range(somoc):
        X.append(float(input(f"Nhập X{i+1}: ")))
    X = np.array([X])
    print('\n')
    return X

def nhapY(somoc):
    Y = []
    for i in range(somoc):
        Y.append(float(input(f"Nhập Y{i+1}: ")))
    Y = np.array([Y])
    return Y


def Binhphuongtoithieu(x, f, phi_x):
    print('Phi(x):\n', pd.DataFrame(phi_x, index=['phi_' + str(i + 1) for i in range(len(phi_x))]))

    heso = np.dot(phi_x, phi_x.T)
    print('\nBộ hệ số:\n', heso)

    heso_tudo = np.dot(phi_x, f.T)
    print('\nHệ số tự do: \n', heso_tudo)

    nghiem = np.dot(np.linalg.inv(heso), heso_tudo)
    print('\nNghiệm của hệ: \n', nghiem)


def Tichphangandung_HT(a, b, h, f, daoham2=None, epxilon=0):
    if epxilon != 0 and daoham2 != None:
        x_1 = np.linspace(a, b, 1000000)
        M2 = np.max(np.abs(daoham2(x_1)))
        print('M2 =', M2)
        h_new = np.sqrt(epxilon * 12 / M2 / (b - a))
        print('h =', h_new)
        x = np.linspace(a, b, int((b - a) / h_new) + 2).reshape(1, -1)
        y = f(x)
        print(pd.DataFrame(np.concatenate([x, y]), index=['x', 'y']))
        Ih = (h_new / 2) * (y[0, 0] + y[0, -1] + 2 * np.sum(y[0, 1:-1]))
        print('\nIh =', Ih)
    else:
        print('h =', h)

        h_2 = h / 2
        print('h / 2 =', h_2, '\n')

        x = np.linspace(a, b, int((b - a) / h) + 1).reshape(1, -1)
        y = f(x)
        print(pd.DataFrame(np.concatenate([x, y]), index=['x', 'y']))

        x_2 = np.linspace(a, b, int((b - a) / (h / 2)) + 1).reshape(1, -1)
        y_2 = f(x_2)
        print('\n', pd.DataFrame(np.concatenate([x_2, y_2]), index=['x(h/2)', 'y(h/2)']))

        Ih = (h / 2) * (y[0, 0] + y[0, -1] + 2 * np.sum(y[0, 1:-1]))
        print('\nIh =', Ih)

        Ih_2 = (h_2 / 2) * (y_2[0, 0] + y_2[0, -1] + 2 * np.sum(y_2[0, 1:-1]))
        print('I_h/2 =', Ih_2)

        if daoham2 != None:
            x_1 = np.linspace(a, b, 1000000)
            M2 = np.max(np.abs(daoham2(x_1)))
            print('M2 =', M2)
            saiso = M2 / 12 * (b - a) * h ** 2
            print("Sai số:", saiso)

        saiso_luoiphu = np.abs(Ih - Ih_2) / 3
        print(f'Sai số qua lưới phủ: {saiso_luoiphu}')


def Simpson(a, b, h, f, daoham4=None, epxilon=0):
    if epxilon != 0 and daoham4 != None:
        x_1 = np.linspace(a, b, 1000000)
        M4 = np.max(np.abs(daoham4(x_1)))
        print('M4 =', M4)
        h_new = np.sqrt(np.sqrt(epxilon * 180/ M4 / (b - a)))
        print('h =', h_new)
        x = np.linspace(a, b, int((b - a) / h_new) + 2).reshape(1, -1)
        y = f(x)
        print(pd.DataFrame(np.concatenate([x, y]), index=['x', 'y']))
        phi1 = np.sum(y[0, 1:-1:2])
        phi2 = np.sum(y[0, 2:-1:2])
        Ih = h_new / 3 * (y[0, 0] + y[0, -1] + 4 * phi1 + 2 * phi2)
        print(f'\nphi1 = {phi1}')
        print(f'phi2 = {phi2}')
        print(f'Ih = {Ih}')
    else:
        print('h =', h)

        h_2 = h / 2
        print('h / 2 =', h_2, '\n')

        x = np.linspace(a, b, int((b - a) / h) + 1).reshape(1, -1)
        y = f(x)
        print(pd.DataFrame(np.concatenate([x, y]), index=['x', 'y']))

        x_2 = np.linspace(a, b, int((b - a) / (h / 2)) + 1).reshape(1, -1)
        y_2 = f(x_2)
        print('\n', pd.DataFrame(np.concatenate([x_2, y_2]), index=['x(h/2)', 'y(h/2)']))

        phi1 = np.sum(y[0, 1:-1:2])
        phi2 = np.sum(y[0, 2:-1:2])
        Ih = h / 3 * (y[0, 0] + y[0, -1] + 4 * phi1 + 2 * phi2)
        print(f'\nphi1 = {phi1}')
        print(f'phi2 = {phi2}')
        print(f'Ih = {Ih}')

        phi1_2 = np.sum(y_2[0, 1:-1:2])
        phi2_2 = np.sum(y_2[0, 2:-1:2])
        Ih_2 = h_2 / 3 * (y_2[0, 0] + y_2[0, -1] + 4 * phi1_2 + 2 * phi2_2)
        print(f'\nphi1 (h/2) = {phi1_2}')
        print(f'phi2 (h/2) = {phi2_2}')
        print('I_h/2 =', Ih_2, '\n')

        if daoham4 != None:
            x_1 = np.linspace(a, b, 1000000)
            M4 = np.max(np.abs(daoham4(x_1)))
            print('M4 =', M4)
            saiso = M4 / 180 * (b - a) * h ** 4
            print("Sai số:", saiso)

        saiso_luoiphu = np.abs(Ih - Ih_2) * 16 / 15
        print(f'Sai số qua lưới phủ: {saiso_luoiphu}')

def Euler_hien(a, b, h, w0, f):
    x = np.linspace(a, b, int((b - a)/h) + 1).reshape(1, -1)
    y = w0.copy()
    for i in range(1, len(x[0])):
        yk = y[:, i - 1].reshape(-1,1) + h * f(x[0, i - 1], y[0, i - 1], y[1, i - 1])
        y = np.concatenate([y, yk], axis=1)
    final = np.concatenate([x, y])
    print(pd.DataFrame(final.T, columns=['x', 'y', 'z']))


def Euler_an(a, b, h, y0, f):
    x = np.linspace(a, b, int((b - a) / h) + 1).reshape(1, -1)
    y = np.zeros(x.shape)
    y[0, 0] = y0

    u = np.zeros(x.shape)
    u[0, 0] = y[0, 0] + h * f(x[0, 0], y[0, 0])

    for i in range(1, len(u[0])):
        y[0, i] = y[0, i - 1] + h * f(x[0, i], u[0, i - 1])
        u[0, i] = y[0, i] + h * f(x[0, i], y[0, i])

    result = np.concatenate([x, y, u], axis=0)
    print(pd.DataFrame(result.T, columns=['x', 'y', 'u']))


def Euler_caitien(a, b, h, y0, f):
    x = np.linspace(a, b, int((b - a) / h) + 1).reshape(1, -1)
    y = np.zeros(x.shape)
    y[0, 0] = y0

    u = np.zeros(x.shape)
    u[0, 0] = y[0, 0] + h * f(x[0, 0], y[0, 0])

    for i in range(1, len(u[0])):
        y[0, i] = y[0, i - 1] + h / 2 * f(x[0, i - 1], y[0, i - 1]) + h / 2 * f(x[0, i], u[0, i - 1])
        u[0, i] = y[0, i] + h * f(x[0, i], y[0, i])

    result = np.concatenate([x, y, u], axis=0)
    print(pd.DataFrame(result.T, columns=['x', 'y', 'u']))


def RK_4(a, b, h, y0, z0, f, g):
    x = np.linspace(a, b, int((b - a) / h) + 1).reshape(1, -1)
    y = np.zeros(x.shape)
    z = np.zeros(x.shape)

    k1 = np.zeros(x.shape)
    k2 = np.zeros(x.shape)
    k3 = np.zeros(x.shape)
    k4 = np.zeros(x.shape)

    l1 = np.zeros(x.shape)
    l2 = np.zeros(x.shape)
    l3 = np.zeros(x.shape)
    l4 = np.zeros(x.shape)

    for i in range(len(x[0])):
        if i == 0:
            y[0, i] = y0
            z[0, i] = z0
        else:
            y[0, i] = y[0, i - 1] + 1 / 6 * (k1[0, i - 1] + 2 * k2[0, i - 1] + 2 * k3[0, i - 1] + k4[0, i - 1])
            z[0, i] = z[0, i - 1] + 1 / 6 * (l1[0, i - 1] + 2 * l2[0, i - 1] + 2 * l3[0, i - 1] + l4[0, i - 1])

        k1[0, i] = h * f(x[0, i], y[0, i], z[0, i])
        l1[0, i] = h * g(x[0, i], y[0, i], z[0, i])

        k2[0, i] = h * f(x[0, i] + h / 2, y[0, i] + 1 / 2 * k1[0, i], z[0, i] + 1 / 2 * l1[0, i])
        l2[0, i] = h * g(x[0, i] + h / 2, y[0, i] + 1 / 2 * k1[0, i], z[0, i] + 1 / 2 * l1[0, i])

        k3[0, i] = h * f(x[0, i] + 1 / 2 * h, y[0, i] + 1 / 2 * k2[0, i], z[0, i] + 1 / 2 * l2[0, i])
        l3[0, i] = h * g(x[0, i] + 1 / 2 * h, y[0, i] + 1 / 2 * k2[0, i], z[0, i] + 1 / 2 * l2[0, i])

        k4[0, i] = h * f(x[0, i] + h, y[0, i] + k3[0, i], z[0, i] + l3[0, i])
        l4[0, i] = h * g(x[0, i] + h, y[0, i] + k3[0, i], z[0, i] + l3[0, i])

    result = np.concatenate([x, y, z, k1, l1, k2, l2, k3, l3, k4, l4])
    print(pd.DataFrame(result.T, columns=['x', 'y', 'z', 'k1', 'l1', 'k2', 'l2', 'k3', 'l3', 'k4', 'l4']))