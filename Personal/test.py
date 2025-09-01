def minPathSum(grid) -> int:
        n = len(grid)
        m = len(grid[0])
        dp = [[0 for _ in range(m)] for __ in range(n)]

        dp[0][0] = grid[0][0]

        for j in range(1, m):
            dp[0][j] = grid[0][j] + dp[0][j-1]

        for i in range(1, n):
            dp[i][0] = grid[i][0] + dp[i-1][0]

        for i in range(1,n):
            for j in range(1,m):
                dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
        return dp[n-1][m-1]

grid = [[1,2,3],[4,5,6]]

print(minPathSum(grid))