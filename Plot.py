import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Final_Acce.csv")
df = df.drop(['Index','Y','Z'], 1)
df1 = df[df.Activity==3]
df1=df1.iloc[:200,0]
df2 = df[df.Activity==4]
df2=df2.iloc[:200,0]
df3 = df[df.Activity==5]
df3=df3.iloc[:200,0]
out=[]

for i in range(200):
    out.append(i)
plt.plot(out,df1,label='Standing')
plt.plot(out,df2,label='Walking')
plt.plot(out,df3,label='Going Up\Down Stairs')
plt.xlabel('Time in ms')
plt.ylabel('Accelerometer X axis Measurement')
plt.title("3 different activity Accelerometer X axis Measurement")
plt.legend()
plt.show()

df = pd.read_csv("Missing.csv")
plt.title("Missing Prediction")
plt.xlabel("Time in ms")
plt.ylabel("Accelerometer X-axis measurement")
plt.plot(df.iloc[:1400],label="Actual value",color='green')
plt.plot(df.iloc[1400:1600],label="Missing value")
plt.plot(df.iloc[1600:],label="Actual value",color='green')
plt.legend()
plt.show()