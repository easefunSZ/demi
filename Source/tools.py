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
    for key, value in pairs(table):

        val_string = ''

        if type(value) == 'table':
            val_string = self:table_to_string(value)
        else:
            val_string = tostring(value)

        out = out + tostring(key) + ":" + val_string + ", "

    out = out + "}"
    return out


'''
--- An arbitrarily large number used for clamping regrets.
--@return the number
'''


def M: max_number()

:
return 999999
return M
