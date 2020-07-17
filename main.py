from LinearReg import LinearReg
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import re
import pandas as pd
from tkinter import messagebox
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
style.use('ggplot')
import threading
import numpy as np

class InitializeWindow:

    def __init__(self, window):

        self.window = window
        self.window.geometry('600x600')
        self.window.resizable(width=False, height=False)
        self.window.title('EAGLE-AI')

        self.train_filename = None
        self.train_df = pd.DataFrame({'P' : []}) #empty dataframe

        self.crossval_filename = None
        self.crossval_df = pd.DataFrame({'P' : []}) #empty dataframe

        self.test_filename = None
        self.test_df = pd.DataFrame({'P' : []}) #empty dataframe

        self.validate = False

        self.frame1 = tk.Frame(master=self.window,relief=tk.SUNKEN, width=600, height=400, bg='#cbcaca')
        self.frame1.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

        self.frame2 = tk.Frame(master=self.window,relief=tk.SUNKEN , width=600, height=200, bg="black")
        self.frame2.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)

        #Upper Frame
        self.image = Image.open("eagle9.png")
        self.backgroundImage=ImageTk.PhotoImage(self.image.resize((600,400)))
        self.label = tk.Label(master=self.frame1, image = self.backgroundImage)
        self.label.pack()

        self.btn_ini = tk.Button(
        master=self.frame1,
        text="INITIALIZE",
        command = self.validate_initialize,
        bg="green",
        fg="white",
        )
        self.btn_ini.pack()

        #Lower Frame
        self.btn_train = tk.Button(
            master=self.frame2,
            text="LOAD TRAINING SET",
            command=lambda: self.load('train'),
            bg="blue",
            fg="white",
        )
        self.btn_train.pack(pady=8)

        self.btn_val = tk.Button(
            master=self.frame2,
            text="LOAD CrossValidation/DEV SET",
            command=lambda: self.load('val'),
            bg="blue",
            fg="white",
        )
        self.btn_val.pack(pady=8)

        self.btn_test = tk.Button(
            master=self.frame2,
            text="LOAD TEST SET",
            command=lambda: self.load('test'),
            bg="blue",
            fg="white",
        )
        self.btn_test.pack(pady=8)

    def load(self, file_type):

        name = askopenfilename(filetypes=[('CSV', '*.csv',), ('Excel', ('*.xls', '*.xlsx'))])

        if name:
            if name.endswith('.csv'):
                if file_type == 'train':
                    self.train_df = pd.read_csv(name)
                if file_type == 'val':
                    self.crossval_df = pd.read_csv(name)
                if file_type == 'test':
                    self.test_df = pd.read_csv(name)
            else:
                if file_type == 'train':
                    self.train_df = pd.read_excel(name)
                if file_type == 'val':
                    self.crossval_df = pd.read_excel(name)
                if file_type == 'test':
                    self.test_df = pd.read_excel(name)

            if file_type == 'train':
                self.train_filename = name
                self.btn_train['text'] = self.get_filename(self.train_filename)
            if file_type == 'val':
                self.crossval_filename = name
                self.btn_val['text'] = self.get_filename(self.crossval_filename)
            if file_type == 'test':
                self.test_filename = name
                self.btn_test['text'] = self.get_filename(self.test_filename)

    def get_filename(self,location):

        file_list = location.split('/')
        return file_list[len(file_list)-1]

    def validate_initialize(self):
        if self.train_df.empty or self.crossval_df.empty or self.train_df.empty:
            messagebox.showerror('Error', 'All DataSets Must be Loaded')
        else:
           self.validate = True
           self.window.destroy()



class TrainWindow:
    def __init__(self, window, train_df, val_df, test_df):
        self.window = window
        self.window.geometry('1300x600')
        self.window.resizable(width=False, height=False)
        self.window.title("EAGLE-AI")

        self.epoch_entry = None
        self.alpha_entry = None
        self.lambda_entry = None

        self.epoch_list = []
        self.cost_list = []

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.frame1 = tk.Frame(master=self.window,relief=tk.SUNKEN, width=200, height=600)
        self.frame1.grid(row=0, column=0, padx=8)
        self.frame2 = tk.Frame(master=self.window,relief=tk.SUNKEN, width=800, height=600)
        self.frame2.grid(row=0, column=1)
        self.frame3 = tk.Frame(master=self.window,relief=tk.SUNKEN, width=300, height=600)
        self.frame3.grid(row=0, column=2, padx=8)

        self.frame4 = tk.Frame(master=self.frame2,relief=tk.SUNKEN, width=800, height=200)
        self.frame4.grid(row=0, column=0)
        self.frame5 = tk.Frame(master=self.frame2,relief=tk.SUNKEN, width=800, height=400)
        self.frame5.grid(row=1, column=0)

        #frame 4
        self.select_label = tk.Label(master=self.frame4, text="Select The Algorithm:")
        self.select_label.config(font=("Courier", 20))
        self.select_label.pack()
        self.algos = ttk.Combobox(master=self.frame4,
                                    height=40,
                                    
                                    values=['Linear Regression', 'Logistic Regression'],
                                    state='readonly',
                                    )
        self.algos.bind("<<ComboboxSelected>>", self.load_gui)
        self.algos.pack(pady=8)
        self.start_btn = tk.Button(master=self.frame4, text="START TRAINIG", bg='blue', fg='white', command=self.start_train)
        self.start_btn.pack()

        #frame 5
        self.f = Figure(figsize=(5,4), dpi=100)
        self.ax = self.f.add_subplot(111)
       
       
        

        self.ax.title.set_text('Cost Function')
        self.ax.set_ylabel('Cost')
        self.ax.set_xlabel('iterations')
        self.canvas = FigureCanvasTkAgg(self.f, self.frame5)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=8)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
        self.progress = ttk.Progressbar(master=self.frame5,
                                        style="green.Horizontal.TProgressbar",
                                        orient=HORIZONTAL, 
                                        mode='determinate',
                                        length=100)
        self.progress.pack(pady=8)

        
        
        

    def load_gui(self, event):
        #frame 1
        self.frame1.rowconfigure(0,weight=3,minsize=100)
        self.frame1.rowconfigure(1,weight=1,minsize=50)
        self.frame1.rowconfigure(2,weight=1,minsize=50)
        self.frame1.rowconfigure(3,weight=1,minsize=50)
        self.frame1.rowconfigure(4,weight=1,minsize=50)
        self.frame1.rowconfigure(5,weight=1,minsize=50)
        self.frame1.rowconfigure(6,weight=1,minsize=50)
        if self.algos.get() == 'Linear Regression':
            frame_label = tk.Label(master=self.frame1, text="Hyperparameters=>")
            frame_label.config(font=("Courier", 20))
            frame_label.grid(row=0,column=0, sticky='n')

            epoch_label = tk.Label(master=self.frame1, text="Epoch:")
            epoch_label.config(font=(10))
            epoch_label.grid(row=1,column=0)
            self.epoch_entry = tk.Entry(master=self.frame1)
            self.epoch_entry.grid(row=2,column=0)

            alpha_label = tk.Label(master=self.frame1, text="Learning Rate:")
            alpha_label.config(font=(10))
            alpha_label.grid(row=3,column=0)
            self.alpha_entry = tk.Entry(master=self.frame1)
            self.alpha_entry.grid(row=4,column=0)

            lambda_label = tk.Label(master=self.frame1, text="Lambda(regularization):")
            lambda_label.config(font=(10))
            lambda_label.grid(row=5,column=0)
            self.lambda_entry= tk.Entry(master=self.frame1)
            self.lambda_entry.grid(row=6,column=0)


        #frame 3
        self.frame3.rowconfigure(0,weight=3,minsize=100)
        self.frame3.rowconfigure(1,weight=1,minsize=50)
        self.frame3.rowconfigure(2,weight=1,minsize=50)
        self.frame3.rowconfigure(3,weight=1,minsize=50)
        if self.algos.get() == 'Linear Regression':
            frame3_label = tk.Label(master=self.frame3, text="Performance Analysis=>")
            frame3_label.config(font=("Courier", 20))
            frame3_label.grid(row=0,column=0, sticky='n')

            self.r2_label = tk.Label(master=self.frame3, text="R-Squared:")
            self.r2_label.config(font=(10))
            self.r2_label.grid(row=1,column=0)

            self.rmse_label = tk.Label(master=self.frame3, text="Root Mean Squared Error:")
            self.rmse_label.config(font=(10))
            self.rmse_label.grid(row=2,column=0)

            self.mse_label = tk.Label(master=self.frame3, text="Mean Squared Error:")
            self.mse_label.config(font=(10))
            self.mse_label.grid(row=3,column=0)
          


    

    def animate(self, i):
        if len(self.epoch_list)!=0 and len(self.cost_list)!=0:
            self.ax.clear()
            self.ax.title.set_text('Cost Function')
            self.ax.set_ylabel('Cost')
            self.ax.set_xlabel('iterations')
            self.ax.plot(self.epoch_list, self.cost_list)
            
        
        


    def start_train(self):
        flag = 0
        try:
            algo = self.algos.get()
            if algo == 'Linear Regression':
                linear_reg = LinearReg()
                linear_reg.train(self.train_df, self.test_df)
        
                epoch = int(self.epoch_entry.get())
                alpha = float(self.alpha_entry.get())
                lambd = float(self.lambda_entry.get())

                self.cost_list = []
                self.epoch_list = []

                for t in range(1,epoch+1):
                
                    self.cost_list.append(linear_reg._cal_cost(linear_reg.train_x, linear_reg.train_y, linear_reg.theta, lambd))
                    self.epoch_list.append(t)
                    linear_reg.theta = linear_reg.theta - alpha*linear_reg._gradient(linear_reg.train_x, linear_reg.train_y, linear_reg.theta, lambd)
                    self.progress['value'] = int((t/epoch)*100)
                    self.frame5.update_idletasks() 

                y_predic = np.dot(linear_reg.test_x, linear_reg.theta)
                y_predic = pd.DataFrame(y_predic)
                
                flag=1

                r2 = linear_reg.cal_r2(linear_reg.test_y, y_predic)
                self.r2_label['text'] = f"R-Squared:{r2}"

                rmse = linear_reg.cal_rmse(linear_reg.test_y, y_predic)
                self.rmse_label['text'] = f"Root Mean Squared Error:{rmse}"

                mse = linear_reg.cal_mse(linear_reg.test_y, y_predic)
                self.mse_label['text'] = f"Mean Squared Error:{mse}"
                
            messagebox.showinfo('Success','Model Trained Successfully!!')
        except:
            if flag == 0:
                messagebox.showerror('Error', 'Model Training Failed(Check your hyperparameters and Input Files)')
                self.cost_list = []
                self.epoch_list = []
                self.r2_label['text'] = "R-Squared:Failed"
                self.rmse_label['text'] = "Root Mean Squared Error:Failed"
                self.mse_label['text'] = "Mean Squared Error:Failed"
            else:
                messagebox.showerror('Error', 'Model Performace Analysis Failed')
                self.r2_label['text'] = "R-Squared:Failed"
                self.rmse_label['text'] = "Root Mean Squared Error:Failed"
                self.mse_label['text'] = "Mean Squared Error:Failed"


# --- main ---

if __name__ == '__main__':
    root = tk.Tk()
    ini = InitializeWindow(root)
    root.mainloop()
    if ini.validate == True:
        root2 = tk.Tk()
        train = TrainWindow(root2, ini.train_df, ini.crossval_df, ini.test_df)
        anim = animation.FuncAnimation(train.f, train.animate, interval=1000)
        root2.mainloop()