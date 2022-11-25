class BatchIterator:
    def __init__(
        self,
        x,
        y,
        batch_size,
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr >= len(self.x):
            raise StopIteration()

        res = (
            self.x[self.ptr : self.ptr + self.batch_size],
            self.y[self.ptr : self.ptr + self.batch_size],
        )
        self.ptr += self.batch_size
        return res
