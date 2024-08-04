#find the factorial of a number.
num=int(input("Please enter the number that you want to find the factorial of \n"))

fact=1
for i in range(1,num+1):
    fact*=i

print(f"The factorial of the {num} is {fact}")