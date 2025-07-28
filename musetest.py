import muselsl

muse = muselsl.list_muses()

if muse:
    print(muse)
    selmuse = input('뮤즈를 선택하세요')
    while True:
        if int(selmuse) in muse:
            break
        else:
            print("'" + selmuse + "'" + " 라는 이름의 뮤즈는 존재하지 않습니다.")
            selmuse = input('다시 선택하세요')