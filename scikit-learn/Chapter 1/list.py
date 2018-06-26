# -*- coding: utf-8 -*-
class testOne(object):
    def phonelist(self, phone):
        phoneList = [];
        list = phone[0:7]
        lista = int(phone[7:11]) + 1

        for i in range(1000, 1005):
            lista += 1
            listb = list + str(lista)
            phoneList.append(listb)
        return phoneList

count = testOne()
phone = '17711361000'
phoneList = count.phonelist(phone)
for i in phoneList:
    print(i)


