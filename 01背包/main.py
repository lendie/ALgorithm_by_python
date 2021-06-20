# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
#这是01背包问题的求解python代码。

def bag(n,c,w,v):#n表示物品数量，c表示背包所能够容纳的数量，w表示每个物品的重量，v表示每个物品的价值。
    dp = [[0 for i in range(c+1)] for j in range(n+1)]#这里创建了一个二维数组，且初始化都为0
    for i in range(1,n+1):
        for j in range(1,c+1):
            if j<w[i-1]:
                dp[i][j] = dp[i-1][j]
            elif j>=w[i-1]:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i-1]]+v[i-1])
    for x in dp:
        print(x)
    return dp
#01背包问题时间复杂度为O(cn)，空间复杂度为O(cn)，但是空间复杂度可以降低为O（c）.
#分析可得，每一次的计算只用到了前一次的结果，那么前前几次的结果就可以不用存储啦。

def bag1(n,c,w,v):
    dp = [0 for i in range(c+1)]
    for i in range(1,n+1):
        for j in range(c,0,-1):
            if j >= w[i-1]:
                dp[j] = max(dp[j-w[i-1]] + v[i-1], dp[j])
    print(dp)


def show_info(n,c,w,v):
    x = [False for i in range(n)]
    j = c
    for i in range(n,0,-1):
        if v[i][j] > v[i-1][j]:
            x[i-1] = True
            j -= w[i-1]
    print("背包所装物品为：")
    for i in range(n):
        if x[i]:
            print('第', i+1, '个 ', )

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    n = 6
    c = 10
    w = [2,2,3,1,5,2]
    v = [2,3,1,5,4,3]
    dp = bag(n,c,w,v)
    show_info(n,c,w,dp)
    print("这是优化了空间复杂度的结果：")
    bag1(n,c,w,v)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
