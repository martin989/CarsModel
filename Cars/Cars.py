import tkinter as tk
from tkinter.ttk import Combobox
from csv import reader
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing


#Create User interface
class CreateUI:
    def __init__(self, window,filename):
        self.filename = filename
        #get dataset information for creating combo boxes
        self.dataset = DataSet(filename)
        self.lbl0=tk.Label(window, text='Enter information below to calculate the value of the vehicle:')
        self.lbl0.grid(row=0, column=1)

        #Create label and combobox for title status
        self.lbl1=tk.Label(window, text='Title Status Name')
        self.lbl1.grid(row=1, column=0)
        self.cb1=Combobox(window, values=[x for x in self.dataset.statuslist])
        self.cb1.grid(row=1, column=1)

        #Create label and combobox for make
        self.lbl2=tk.Label(window, text='Make Of Car')
        self.lbl2.grid(row=2, column=0)
        self.cb2=Combobox(window, values=[x for x in self.dataset.brandlist])
        self.cb2.grid(row=2, column=1)

        #Create label and combobox for model
        self.lbl3=tk.Label(window, text='Model Select')
        self.lbl3.grid(row=3, column=0)  
        self.cb3=Combobox(window, values=[x for x in self.dataset.modellist])
        self.cb3.grid(row=3, column=1)
         
        #Mileage Entry box
        self.lbl4=tk.Label(window, text='Mileage')
        self.lbl4.grid(row=4, column=0)  
        self.mileageEntry = tk.Entry(window)
        self.mileageEntry.grid(row=4, column=1)

        #Year Entry box
        self.lbl5=tk.Label(window, text='Year')
        self.lbl5.grid(row=5, column=0)  
        self.yearEntry = tk.Entry(window)
        self.yearEntry.grid(row=5, column=1)

        #Create label and combobox for model
        self.lbl6=tk.Label(window, text='Color')
        self.lbl6.grid(row=6, column=0)  
        self.cb4=Combobox(window, values=[x for x in self.dataset.colorlist])        
        self.cb4.grid(row=6, column=1)

        #Create label and combobox for state
        self.lbl7=tk.Label(window, text='State')
        self.lbl7.grid(row=7, column=0)  
        self.cb5=Combobox(window, values=[x for x in self.dataset.statelist])        
        self.cb5.grid(row=7, column=1)

        #Add Button to find result
        self.lbl8=tk.Label(window, text='')
        self.lbl8.grid(row=8, column=1)
        self.b1=tk.Button(window, text='Find Result', command=self.getResult)
        self.b1.grid(row=8, column=0)
        
        #Train Button to train model
        self.lbl8=tk.Label(window, text='')
        self.lbl8.grid(row=15, column=1)
        self.b2=tk.Button(window, text='Train', command=self.getTrain)
        self.b2.grid(row=14, column=0)

    def getResult(self): 
        _titlestatus = str(self.cb1.get())
        _make = str(self.cb2.get())
        _model = str(self.cb3.get()) 
        _mileage = str(self.mileageEntry.get())
        _year = str(self.yearEntry.get())
        _color = str(self.cb4.get())
        _state = str(self.cb5.get())
        model = joblib.load('cars.pkl')
        le = preprocessing.LabelEncoder()
        car = [_make,_model,_year,_titlestatus,_mileage,_color,_state,'usa']
        le.fit([_make,_model,_year,_titlestatus,_mileage,_color,_state,'usa'])
        car = le.transform(car)
        #car = [_make,_model,_year,_titlestatus,_mileage,_color,_state,'usa']
        cars = [car]
        #labelencoder = preprocessing.LabelEncoder()
        #cars = cars.apply(labelencoder.fit_transform)
        car_value = model.predict(cars)
        predicted_value = str(abs(round(car_value[0], 2)))
        self.lbl8['text']='Predicted_value: $' + predicted_value

    def getTrain(self):
        # Load our data set
        df = pd.read_csv(self.filename)
        
        #Handle categorical data
        labelencoder = preprocessing.LabelEncoder()
        df = df.apply(labelencoder.fit_transform)

        # Create the X and y arrays
        X = df[["brand","model","year","title_status","mileage","color","state","country"]]
        y = df["price"]
        # Split the data set in a training set (75%) and a test set (25%)
        _percent = .25
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_percent)

        # Create the Linear Regression model
        model = LinearRegression()
     
        # Train the model
        model.fit(X_train, y_train)

        # Save the trained model to a file so we can use it to make predictions later
        joblib.dump(model, 'cars.pkl')

        # Report an error rate on the training set
        mse_train = mean_absolute_error(y_train, model.predict(X_train))
        self.lbl8['text']='Training Set Error: ' + str(mse_train)

        # Report an error rate on the test set
        mse_test = mean_absolute_error(y_test, model.predict(X_test))    

 #Helper to define combo boxes in UI
class DataSet():
    def __init__(self,filename):
        self.dataset = []
        self.brandlist = []
        self.modellist = []
        self.statuslist = []
        self.statelist = []
        self.colorlist = []
        with open(filename, 'r') as read_obj:
            csv_reader = reader(read_obj)
            for row in csv_reader:
                self.dataset.append(row)
        #get all 
        list = [item[1] for item in self.dataset]
        listarray = np.array(list)
        self.brandlist = np.unique(listarray)
        list = [item[2] for item in self.dataset]
        listarray = np.array(list)
        self.modellist = np.unique(listarray)
        list = [item[4] for item in self.dataset]
        listarray = np.array(list)
        self.statuslist = np.unique(listarray)
        list = [item[4] for item in self.dataset]
        listarray = np.array(list)
        self.statuslist = np.unique(listarray)
        list = [item[6] for item in self.dataset]
        listarray = np.array(list)
        self.colorlist = np.unique(listarray)
        list = [item[7] for item in self.dataset]
        listarray = np.array(list)
        self.statelist = np.unique(listarray)



if __name__ == '__main__':
    filename='C:\\Users\\erinm\\Desktop\\Martin School\\SENG-309 Summer 2020\\A6\\Homework\\CarsDataset.csv'
    window=tk.Tk()
    uiwindow=CreateUI(window,filename)
    window.title('Car Value')
    window.geometry("450x300+10+10")
    window.mainloop()