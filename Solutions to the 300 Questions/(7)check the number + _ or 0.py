#check if a number is positive, negative or zero.

num=int(input("Enter the number that you want to check whether the number is positive, negative or Zero\n"))
if(num>0):
    res="positive number"
elif(num<0):
    res="Negative Number"
else:
    res="Zero"

print(f"The number that you have entered {num} is a {res}")
