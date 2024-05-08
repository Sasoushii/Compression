n=500
i=0
sum=""
while n>0:
    sum+=str(n%2)
    n=n//2
    i+=1
sum=sum[::-1]
print(sum)