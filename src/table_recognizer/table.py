class Table:
    def __init__(self, img, rows):
        self.img = img
        self.rows = rows

    def column_data(self, column_num):
        """
        Return only data rows from table.

        @param column_num: column number
        @return: return rows without headers
        """
        rows = self.column_rows(column_num)

        return rows[3:]

    def column_rows(self, column_num):
        """
        Get list of cells from table

        >>> get_column_rows(3)
        # [[(0,0), (10,0), (0,10), (10,10)], ...]

        Args:
            column_num: the column number

        Returns:
            a list of cells of selected column
        """
        column = []
        for row in self.rows:
            column.append(row[column_num:column_num + 2])

        column_cell_position = []
        column_iter = iter(column)
        prev = next(column_iter)

        for row in column:
            column_cell_position.append(row + prev)
            prev = row

        return column_cell_position
