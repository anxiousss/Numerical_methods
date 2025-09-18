from typing import List
from itertools import combinations



class Matrix:
    def __init__(self, rows: int, columns: int, matrix:List[List[int]] = None,
                 free_members: List[int] = None, diagonals: List[List[int]] = None):
        if diagonals is not None:
            self.diagonals = diagonals

        self.rows: int = rows
        self.columns: int = columns
        self.matrix: List[List[int]] = matrix
        self.free_members = free_members
        self.determinant = self._calc_determinant()

    def transpose(self):
        diff = abs(self.rows - self.columns)
        if max(self.rows, self.columns) == self.rows and diff != 0:
            extra = [[] for _ in range(diff + 1)]
            for j in range(self.columns):
                for i in range(self.rows):
                    extra[j].append(self.matrix[i][j])
            self.matrix.clear()
            self.matrix = extra
            self.rows, self.columns = self.columns, self.rows
            return

        for i, j in list(combinations([_ for _ in range(min(self.rows, self.columns))], 2)):
            self.matrix[i][j], self.matrix[j][i] = self.matrix[j][i], self.matrix[i][j]

        if diff == 0:
            return

        if max(self.rows, self.columns) == self.columns:
            extra = [[] for _ in range(diff)]

            for i in range(self.rows):
                place = 0
                for k in range(self.columns - diff, self.columns):
                    extra[place].append(self.matrix[i][k])
                    place += 1
                self.matrix[i] = self.matrix[i][:self.columns - diff]

            self.matrix.extend(extra)
            self.rows, self.columns = self.columns, self.rows



    def __str__(self):
        return "\n".join([" ".join(map(str, row)) for row in self.matrix])

    def _calc_determinant(self):
        if self.rows == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        else:
            result = 0
            for k in range(self.rows):
                minor_matrix = []
                coefficient = self.matrix[k][0]
                for i in range(self.rows):
                    for j in range(self.columns):
                        if j != 0 and k != i:
                            minor_matrix.append(self.matrix[i][j])
                minor_matrix = [minor_matrix[l:l + self.rows - 1] for l in range(0, len(minor_matrix), self.rows - 1)]
                minor = Matrix(self.rows - 1, self.columns - 1, minor_matrix)
                result += (-1) ** k * coefficient * minor._calc_determinant()

        return result


matrix = Matrix(4, 4, [[2, -3, -4, -1],
                                         [3, 2 ,1, -5],
                                         [-6, 3, -5, 2],
                                         [7, 5, -4, 2]])

print(matrix.determinant)