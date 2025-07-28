testlist = [1, 2, 3, 4, 5]
print(testlist)
selected = input('select')
if not int(selected) in testlist:
    print('error')