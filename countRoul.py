myList = list(range(0,37))
evenCount = 0
oddCount = 0

for x in myList:
	if x % 2 == 0:
		evenCount += 1
	else:
		oddCount += 1

print("Number of evens :", evenCount)
print("Number of odds :", oddCount)

