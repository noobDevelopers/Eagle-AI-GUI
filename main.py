from LinearReg import LinearReg
from LogisticReg import LogisticReg
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

        self.linear_model = None
        self.logistic_model = None

        self.thresh = None

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
        self.save_btn = tk.Button(master=self.frame5, text="SAVE MODEL", bg='green', fg='white', command=self.save_model)
        self.save_btn.pack(pady=8)
        self.save_btn['state'] = "disabled"

        
        
        
    def load_gui(self, event):
        if self.algos.get() == 'Linear Regression':
            self.load_gui_linear()

        if self.algos.get() == 'Logistic Regression':
            self.load_gui_logistic()

    def clear(self):
        self.epoch_entry = None
        self.alpha_entry = None
        self.lambda_entry = None

        self.epoch_list = []
        self.cost_list = []

        self.linear_model = None
        self.logistic_model = None

        self.save_btn['state'] = "disabled"

        self.thresh = None

        for widget in self.frame1.winfo_children():
            widget.destroy()

        for widget in self.frame3.winfo_children():
            widget.destroy()

    def load_gui_linear(self):
        
        self.clear()
        #frame 1
        self.frame1.rowconfigure(0,weight=3,minsize=100)
        self.frame1.rowconfigure(1,weight=1,minsize=50)
        self.frame1.rowconfigure(2,weight=1,minsize=50)
        self.frame1.rowconfigure(3,weight=1,minsize=50)
        self.frame1.rowconfigure(4,weight=1,minsize=50)
        self.frame1.rowconfigure(5,weight=1,minsize=50)
        self.frame1.rowconfigure(6,weight=1,minsize=50)
        # if self.algos.get() == 'Linear Regression':
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
        # if self.algos.get() == 'Linear Regression':
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

    def load_gui_logistic(self):

        self.clear()
        #frame 1
        self.frame1.rowconfigure(0,weight=3,minsize=100)
        self.frame1.rowconfigure(1,weight=1,minsize=50)
        self.frame1.rowconfigure(2,weight=1,minsize=50)
        self.frame1.rowconfigure(3,weight=1,minsize=50)
        self.frame1.rowconfigure(4,weight=1,minsize=50)
        self.frame1.rowconfigure(5,weight=1,minsize=50)
        self.frame1.rowconfigure(6,weight=1,minsize=50)
        self.frame1.rowconfigure(7,weight=1,minsize=50)
        self.frame1.rowconfigure(8,weight=1,minsize=50)
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

        thresh_label = tk.Label(master=self.frame1, text="Threshold:")
        thresh_label.config(font=(10))
        thresh_label.grid(row=7,column=0)
        self.thresh = tk.Entry(master=self.frame1)
        self.thresh.grid(row=8,column=0)

        #frame 3
        self.frame3.rowconfigure(0,weight=3,minsize=100)
        self.frame3.rowconfigure(1,weight=1,minsize=50)
        self.frame3.rowconfigure(2,weight=1,minsize=50)
        self.frame3.rowconfigure(3,weight=1,minsize=50)
        self.frame3.rowconfigure(4,weight=1,minsize=50)
        self.frame3.rowconfigure(5,weight=1,minsize=50)
        # self.frame3.rowconfigure(3,weight=1,minsize=50)
        frame3_label = tk.Label(master=self.frame3, text="Performance Analysis=>")
        frame3_label.config(font=("Courier", 20))
        frame3_label.grid(row=0,column=0, sticky='n')

        self.trainacc_label = tk.Label(master=self.frame3, text="Train Accuracy:")
        self.trainacc_label.config(font=(10))
        self.trainacc_label.grid(row=1,column=0)

        self.testacc_label = tk.Label(master=self.frame3, text="Test Accuracy:")
        self.testacc_label.config(font=(10))
        self.testacc_label.grid(row=2,column=0)

        self.prec_label = tk.Label(master=self.frame3, text="Precision(Test set):")
        self.prec_label.config(font=(10))
        self.prec_label.grid(row=3,column=0)

        self.recall_label = tk.Label(master=self.frame3, text="Recall(Test set):")
        self.recall_label.config(font=(10))
        self.recall_label.grid(row=4,column=0)

        self.f1_label = tk.Label(master=self.frame3, text="F1 Score(Test set):")
        self.f1_label.config(font=(10))
        self.f1_label.grid(row=5,column=0)



    def animate(self, i):
        if len(self.epoch_list)!=0 and len(self.cost_list)!=0:
            self.ax.clear()
            self.ax.title.set_text('Cost Function')
            self.ax.set_ylabel('Cost')
            self.ax.set_xlabel('iterations')
            self.ax.plot(self.epoch_list, self.cost_list)
            
        
        

    def start_train(self):

        algo = self.algos.get()

        if algo == 'Linear Regression':
            self.start_train_linreg()
        elif algo == 'Logistic Regression':
            self.start_train_logreg()
        
    def start_train_linreg(self):
        self.save_btn['state'] = "disabled"
        flag = 0
        try:
            # algo = self.algos.get()
            # if algo == 'Linear Regression':
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
            # print(linear_reg.theta)
            # print(linear_reg.train_x.head())


            y_predic = np.dot(linear_reg.test_x, linear_reg.theta)
            y_predic = pd.DataFrame(y_predic)
            
            flag=1

            r2 = linear_reg.cal_r2(linear_reg.test_y, y_predic)
            self.r2_label['text'] = f"R-Squared:{r2}"

            rmse = linear_reg.cal_rmse(linear_reg.test_y, y_predic)
            self.rmse_label['text'] = f"Root Mean Squared Error:{rmse}"

            mse = linear_reg.cal_mse(linear_reg.test_y, y_predic)
            self.mse_label['text'] = f"Mean Squared Error:{mse}"

            self.linear_model = linear_reg
                
            messagebox.showinfo('Success','Model Trained Successfully!!')
            self.save_btn['state'] = "normal"
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
                self.save_btn['state'] = "normal"

    def start_train_logreg(self):

        self.save_btn['state'] = "disabled"
        # flag = 0
        try:

            logistic_reg = LogisticReg(self.train_df, self.test_df)
            self.logistic_model = logistic_reg
            
            epoch = int(self.epoch_entry.get())
            alpha = float(self.alpha_entry.get())
            lambd = float(self.lambda_entry.get())
            thresh = float(self.thresh.get())

            self.cost_list = []
            self.epoch_list = []

            for t in range(1,epoch+1):
            
                self.cost_list.append(logistic_reg.cal_cost(logistic_reg.train_x, logistic_reg.train_y, logistic_reg.theta, lambd))
                self.epoch_list.append(t)
                logistic_reg.theta = logistic_reg.theta - alpha*(logistic_reg.cal_grad(logistic_reg.train_x,logistic_reg.train_y,logistic_reg.theta,lambd))
                self.progress['value'] = int((t/epoch)*100)
                self.frame5.update_idletasks() 
            # print(logistic_reg.theta)
            # print(logistic_reg.train_x.head())
            ypred_train = logistic_reg.predict(logistic_reg.train_x, thresh)
            ypred_test = logistic_reg.predict(logistic_reg.test_x, thresh)
            accuracy_train = 100 - np.mean(np.abs(ypred_train - logistic_reg.train_y)) * 100
            accuracy_test = 100 - np.mean(np.abs(ypred_test - logistic_reg.test_y)) * 100

            self.testacc_label['text'] = f'Test Accuracy:{np.squeeze(accuracy_test)}'
            self.trainacc_label['text'] = f'Train Accuracy:{np.squeeze(accuracy_train)}'
            
            prec_test,recall_test = logistic_reg.prec_recall(logistic_reg.test_y, ypred_test)
            f1_score = logistic_reg.cal_f1(prec_test, recall_test)

            self.prec_label['text'] = f'Precision(Test set):{prec_test}'
            self.recall_label['text'] = f'Recall(Test set):{recall_test}'
            self.f1_label['text'] = f'F1 Score(Test set):{f1_score}'
            
            messagebox.showinfo('Success','Model Trained Successfully!!')
            self.save_btn['state'] = "normal"
        except:

            messagebox.showerror('Error', 'Model Training Failed(Check your hyperparameters and Input Files)')
            self.cost_list = []
            self.epoch_list = []

        
                     
           


    def get_filename(self,location):

        file_list = location.split('/')
        return file_list[len(file_list)-1]

    def save_model(self):
        try:
            if self.algos.get() == 'Linear Regression':
                export_file_path = tk.filedialog.asksaveasfilename(defaultextension='.csv')
                if(len(export_file_path)!=0):
                    file_name = self.get_filename(export_file_path)
                    second_file = export_file_path[0:export_file_path.find(file_name)]
                    second_file = second_file+f'normalization(mean)_{file_name}'
                    third_file = export_file_path[0:export_file_path.find(file_name)]
                    third_file = third_file+f'normalization(std)_{file_name}'

                    pd.DataFrame(self.linear_model.theta).to_csv(export_file_path, index=False, header=False)
                    self.linear_model._mean.to_csv(second_file,header=False)
                    self.linear_model._std.to_csv(third_file,header=False)
                    

            if self.algos.get() == 'Logistic Regression':
                export_file_path = tk.filedialog.asksaveasfilename(defaultextension='.csv')
                if(len(export_file_path)!=0):
                    file_name = self.get_filename(export_file_path)
                    second_file = export_file_path[0:export_file_path.find(file_name)]
                    second_file = second_file+f'normalization(mean)_{file_name}'
                    third_file = export_file_path[0:export_file_path.find(file_name)]
                    third_file = third_file+f'normalization(std)_{file_name}'
                
                    pd.DataFrame(self.logistic_model.theta).to_csv(export_file_path, index=False, header=False)
                    self.logistic_model._mean.to_csv(second_file,header=False)
                    self.logistic_model._std.to_csv(third_file,header=False)
            messagebox.showinfo('Success','Model Saved Successfully!!')
                
        except:

            messagebox.showerror('Error', 'Failed to save model')
               
                
            
       
        

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