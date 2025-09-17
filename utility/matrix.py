from typing import List
from itertools import combinations



class Matrix:
    def __init__(self, rows: int, columns: int, matrix:List[List[int]] = None, free_members: List[int] = None):
        if matrix is None:
            self.matrix = []
        self.matrix: List[List[int]] = matrix
        self.rows: int = rows
        self.columns: int = columns
        self.free_members = free_members
        #self.determinant

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

    # def _calc_determinant(self):

