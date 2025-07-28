import muselsl

muse = muselsl.list_muses()
muses = len(muse)
musename = []
for i in range (muses):
    musename.append(muse[i]['name'])
if muse:
    print(musename)
    selmuse = input('뮤즈를 선택하세요')
    while True:
        if selmuse in musename:
            break
        else:
            print("'" + str(selmuse) + "'" + ' 라는 이름의 뮤즈는 없습니다.')
            selmuse = input('다시 선택하세요')
    selmuse = musename.index(selmuse)
    muse = muse[selmuse]['address']
    muselsl.stream(muse)
    muselsl.view()