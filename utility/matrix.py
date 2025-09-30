from typing import List


from utility.functions import combinations


class Matrix:
    def __init__(self, rows: int, columns: int, matrix: List[List[int | float]] = None,
                 free_members: List[int | float] = None, diagonals: List[List[int | float]] = None,
                 identity: bool = None):

        self.rows: int = rows
        self.columns: int = columns
        if matrix is None:
            if identity:
                self.matrix = [[int(i == j) for j in range(self.columns)] for i in range(self.rows)]
            else:
                self.matrix = [[0] * self.rows for _ in range(self.rows)]
        else:
            self.matrix: List[List[int | float]] | List[int | float] = matrix
        self.free_members: List[int | float] = free_members
        self.diagonals = diagonals


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

    def calc_determinant(self):
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
                result += (-1) ** k * coefficient * minor.calc_determinant()

        return result

    def _multiply(self, other: "Matrix") -> "Matrix":
        if self.columns != other.rows:
            raise ValueError('Несовместимые размеры матриц.')

        result = Matrix(self.rows, other.columns, None, None, None, False)
        for i in range(self.rows):
            for j in range(other.columns):
                result.matrix[i][j] = sum(self.matrix[i][r] * other.matrix[r][j] for r in range(self.columns))

        return result

    def copy(self):
        return self.__copy__()

    def __mul__(self, other):
        return self._multiply(other)

    def __copy__(self):
        return Matrix(self.rows, self.columns, self.matrix, self.free_members, self.diagonals)
