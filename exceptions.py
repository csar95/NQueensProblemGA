class Error(Exception):

    def __init__(self, n):
        self.n = n


class LessThanFourError(Error):

    def __str__(self):
        return "Number {} is less than 4.".format(self.n)


class IncorrectInputLengthError(Error):

    def __str__(self):
        return "The length of your input is less than {} or characters are not separated by blank spaces.".format(self.n)


class IncorrectInputError(Error):

    def __str__(self):
        return "The input characters are not integers between 0 and {}.".format(self.n)
