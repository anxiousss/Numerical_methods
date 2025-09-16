from typing import List
from itertools import combinations



class Matrix:
    def __init__(self, rows: int, columns: int, matrix:List[List[int]] = None):
        if matrix is None:
            self.matrix = []
        self.matrix: List[List[int]] = matrix
        self.rows: int = rows
        self.columns: int = columns
        #self.determinant

    def transpose(self):
        for i, j in list(combinations([_ for _ in range(min(self.rows, self.columns))], 2)):
            self.matrix[i][j], self.matrix[j][i] = self.matrix[j][i], self.matrix[i][j]
        if self.rows == self.columns:
            return

        diff = abs(self.rows - self.columns)
        if max(self.rows, self.columns) == self.columns:
            extra = [[] for _ in range(diff)]

            for i in range(self.rows):
                for k in range(self.columns - diff, self.columns):
                    extra[k - diff].append(self.matrix[i][k])
                self.matrix[i] = self.matrix[i][:self.columns - diff]

            self.matrix.extend(extra)


    def __str__(self):
        # Кринж
        print(self.matrix)

    # def _calc_determinant(self):


def main():
    matrix = Matrix(2, 4, [[1, 2, 3, 4], [5, 6, 7, 8]])
    matrix.transpose()
    print(matrix)

if __name__ == "__main__":
    main()