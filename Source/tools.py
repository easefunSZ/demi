''''
--- Assorted tools.
--@module tools
'''''
M = {}

'''
--- Generates a string representation of a table.
--@param table the table
--@return the string
'''


def table_to_string(table):
    out = "{"
    for key, value in table.items():

        val_string = ''

        if type(value) == 'table':
            val_string = table_to_string(value)
        else:
            val_string = str(value)

        out = out + str(key) + ":" + val_string + ", "

    out = out + "}"
    return out
'''
--- An arbitrarily large number used for clamping regrets.
--@return the number
'''
def max_number():
    return 999999
