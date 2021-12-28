import numpy as np
from matplotlib import pyplot as plt
import LY_Tools
import random

def polyfit():
    a = np.linspace(1,100,100)
    # print(np.random.randint(0,10))
    # exit()
    b = a ** 2 + np.random.randint(0,500,100)
    # b = a * 2

    plt.plot(a,b)
    # plt.show()
    z = np.polyfit(a,b,2)

    # new_y = a * z[0] + z[1]
    new_y = a**2 * z[0] + z[1]*a + z[2]
    print(z)
    plt.plot(a,new_y,'--')
    plt.show()

def foo():
    x = np.linspace(0,2*np.pi,100)
    # print(np.random.randint(0,10))
    # y :1 sin, 2 ax + b
    k = 0.5
    np.random.seed(1)
    y = np.sin(x) + x * k + np.random.randint(0,200*0.3,len(x))/100.
    part_a_index = np.array([0,1,2])
    part_b_index = np.array([3,4,5,6])
    part_c_index = np.array([7,8,9])

    y_reshape = np.reshape(y,(10,-1))

    part_a_vals = []
    part_b_vals = []
    part_c_vals = []
    part_total_vals = []

    for row in y_reshape:
        part_a_vals_i = LY_Tools.Tools().pick_vals_from_1darray(row,part_a_index)
        part_b_vals_i = LY_Tools.Tools().pick_vals_from_1darray(row,part_b_index)
        part_c_vals_i = LY_Tools.Tools().pick_vals_from_1darray(row,part_c_index)

        part_a_vals.append(np.mean(part_a_vals_i))
        part_b_vals.append(np.mean(part_b_vals_i))
        part_c_vals.append(np.mean(part_c_vals_i))

        part_total_vals.append(np.mean(row))


    x_new = list(range(10))
    T1 = np.polyfit(x_new,part_a_vals,1)
    T2 = np.polyfit(x_new,part_b_vals,1)
    T3 = np.polyfit(x_new,part_c_vals,1)
    T0 = np.polyfit(x_new,part_total_vals,1)

    print(T1)
    print(T2)
    print(T3)
    print(T0)

    plt.plot(part_a_vals,label='a')
    plt.plot(part_b_vals,label='b')
    plt.plot(part_c_vals,label='c')
    plt.plot(part_total_vals,label='total')
    plt.legend()
    plt.show()


    # total_T = np.polyfit(x,y,1)
    # part_a_T = np.polyfit()


def test_random():
    random.seed(1)
    a = np.random.randint(1,1000,15*100)
    x = range(15*100)

    total_start = a[:100]
    total_end = a[-100:]

    part1_start = a[:50]
    part1_end = a[14*100:14*100+50]

    part2_start = a[50:100]
    part2_end = a[14*100+50:14*100+100]


    total_diff = np.mean(total_end)-np.mean(total_start)
    part1_diff = np.mean(part1_end)-np.mean(part1_start)
    part2_diff = np.mean(part2_end)-np.mean(part2_start)
    print(total_diff)
    print(part1_diff)
    print(part2_diff)
    print('*****')
    print(part1_diff + part2_diff)
    # plt.plot(x[:100],total_start)
    # plt.plot(x[-100:],total_end)
    # plt.plot(x[:50],part1_start)
    # plt.plot(x[14*100:14*100+50],part1_end)
    # plt.plot(x[50:100],part2_start)
    # plt.plot(x[14*100+50:14*100+100],part2_end)
    # plt.show()


    # plt.plot(a)
    # plt.show()
    # print(a)


    pass


def test_number():
    # np.random.seed(1)
    a = np.random.randint(0,100,3)
    b = np.random.randint(0,100,3)

    print(a)
    print(b)
    total_diff = np.sum(b) - np.sum(a)
    part1_diff = b[0] - a[0]
    part2_diff = b[1] - a[1]
    part3_diff = b[2] - a[2]

    print('total_diff',total_diff)
    print('part1_diff',part1_diff)
    print('part2_diff',part2_diff)
    print('part3_diff',part3_diff)


    # print(a)
    # print(b)
    pass

def main():

    # test_random()
    test_number()

if __name__ == '__main__':
    main()