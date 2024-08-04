#check if a number is odd or even.
num=int(input("Please enter teh number that you want to check for an even or odd number\n"))
if(num==0):
    res="Neither an even nor an idd number its just Zero"
elif(num%2==0):
    res="Even Number" 
else:
    res="Odd Number"      

print(f"The Number that you have entered is an {res}")